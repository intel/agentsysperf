# AgentSysPerf Storage Layer — Reference for External Consumers

**Audience:** a team outside AgentSysPerf that needs to read our benchmark results
(dashboard, website, leaderboard, notebook) without breaking when we ship a migration.

**Authority:** everything below was verified against the effective schema
(`PRAGMA user_version = 8`) and against the live store at
`$AGENTSYSPERF_HOME/results.db`. Where a planning doc and the shipped
schema disagree, **the code wins and the drift is flagged explicitly** (see §8).
Nothing here is inferred from a plan doc; anything not shipped is labelled
**PLANNED** with its source.

**Row counts in this document are not contractual.** The canonical store is live and
WAL-backed, so every count below moves between one read and the next — an earlier
draft of this file was already stale by 2× within days. Snapshot with
`VACUUM INTO` (never `cp`, §6) and regenerate the numbers yourself:

```sql
-- run against a snapshot; substitute your own table list if the schema moves
SELECT 'runs' t, COUNT(*) n FROM runs
UNION ALL SELECT 'task_results',      COUNT(*) FROM task_results
UNION ALL SELECT 'measurements',      COUNT(*) FROM measurements
UNION ALL SELECT 'spans',             COUNT(*) FROM spans
UNION ALL SELECT 'analyzer_verdicts', COUNT(*) FROM analyzer_verdicts
UNION ALL SELECT 'sweeps',            COUNT(*) FROM sweeps
UNION ALL SELECT 'sweep_points',      COUNT(*) FROM sweep_points
UNION ALL SELECT 'benchmarks',        COUNT(*) FROM benchmarks
UNION ALL SELECT 'artifacts',         COUNT(*) FROM artifacts;
SELECT layer, COUNT(*) FROM measurements GROUP BY layer ORDER BY layer;
```

Measured as of **2026-07-30**, purely to show which tables have *any* data —
expect all of these to grow:

| Table | Rows | Table | Rows |
|---|---|---|---|
| `runs` | 10 | `analyzer_verdicts` | 235 |
| `task_results` | 30 | `benchmarks` | **0** |
| `measurements` | 233 (`l1` 81, `l1_system` 71, `l3` 64, `perfspect` 13, `emon` 4) | `artifacts` | **0** |
| `spans` | 52 (one TB run) | `sweeps` / `sweep_points` | **0** / **0** |

The four empty tables — `sweeps`, `sweep_points`, `benchmarks`, `artifacts` — are not
broken; see §7.6 (*empty-but-valid is the common case*). `spans` is **no longer
empty**: the terminal-bench run `tb_smoke2` carries 52 spans (1 `agent_step`,
30 `llm_call`, 21 `tool_call`), so token/cost panels do have real data for
agent-loop runs even though CLI/synthetic runs still produce none.

---

## 1. Purpose and the three-tier design

The storage layer exists to end a specific failure: measurements used to live only
in per-directory `measurement_records.json` files while the database held run
metadata, so the DB was **not** the system of record for the hardware data. Three
different read paths (dashboards globbing `/tmp`, a JSON→DB round-trip script, a
Prometheus re-push) each saw a different subset. The unification plan's root-cause
table names this "split-brain" and the no-op `store_measurements()` as the actual
risk — *not* the choice of engine
(`docs/STORAGE_UNIFICATION_PLAN.md` §0, §8b).

The design is three tiers with one system of record.

| Tier | Technology | Role | When it kicks in |
|---|---|---|---|
| **System of record** | SQLite, one file at `$AGENTSYSPERF_HOME/results.db`, WAL mode | Every write. Every published number traces here. | Always. Zero-config, no daemon. |
| **Read-side accelerator** | DuckDB, `ATTACH ... (TYPE sqlite, READ_ONLY)` | Wide cross-run / cross-sweep `GROUP BY` that SQLite + `json_extract` is bad at | Optional `[analytics]` extra. Documented crossover ~10–15k rows; below ~300 rows SQLite is ~50× faster. **The crossover was never measured on real data** (§8). |
| **Export / cold tier** | Parquet (`agentsysperf export`) | Publishable bundle, columnar archive, and the DuckDB query target | On demand. Never a second system of record. |

### Why SQLite and not Postgres/ClickHouse

> "SQLite stays the system of record. Already the correct zero-config choice …
> Do NOT adopt ClickHouse/Postgres/S3 — overkill at this scale."
> — `docs/STORAGE_UNIFICATION_PLAN.md` §8b

The research behind that decision observed
that every mature system is a hybrid — Langfuse runs Postgres + ClickHouse + Redis
+ S3 — *but the tiers exist to solve billions-of-rows and OLTP/OLAP contention*,
a scale a single-researcher local benchmark never reaches. Writers here are minutes
apart. WAL gives one writer plus N concurrent readers, which is exactly the
dashboard-reads-while-a-run-writes shape, with no service to operate.

A server backend is a **seam, not a plan**: any non-`sqlite` DSN scheme raises
`NotImplementedError` naming the scheme, and a Postgres backend would ship as an
`agentsysperf.result_stores` entry-point plugin with zero caller changes. It is
trigger-gated on (a) >1 host writing the same store, (b) sustained concurrent
writers beyond WAL's comfort, or (c) per-user isolation/audit
(`docs/STORAGE_IMPLEMENTATION_PLAN.md`, "Database-as-a-service"). **PLANNED.**

### Why DuckDB is additive, never blocking

`duckdb` is imported lazily behind `_duckdb()` with an install hint; the module
imports cleanly without it, and core plus dashboards never import it (pinned by
`test_duckdb_analytics.py::test_core_does_not_import_duckdb_analytics`). `ATTACH`
means zero duplication — DuckDB reads the same SQLite file, so there is no second
copy to drift.

### Why the payload stays JSON

Payload shapes vary wildly: `emon` 3 keys, `l1` 9–10, `l3` 9, `l1_system` 15,
perfspect ~53 (measured). The schema
promotes **six** hot metrics to typed indexed columns and keeps the long tail in
`measurements.payload`:

> "`payload` is the source of truth; promoted columns are a materialized index."
> — `docs/STORAGE_UNIFICATION_PLAN.md` §2

This gives the **promote-when-hot lifecycle**: a new metric lands in JSON with no
migration → becomes a frequent filter → gets an expression index → stabilizes →
`ALTER TABLE ADD COLUMN`. You migrate to *optimize* a metric that earned it, never
to *add* one. (The middle rung, JSON1 expression indexes, is **PLANNED** — zero
exist today, so filtering an unpromoted key is a full scan + `json_extract`.)

### Division of labour with the other systems

| System | Role | Rule |
|---|---|---|
| SQLite store | system of record, incl. tokens and cost | authoritative |
| Filesystem (`$AGENTSYSPERF_HOME/artifacts/`) | bulk evidence (EMON CSVs, plots) | DB stores **pointers only** |
| Prometheus / Grafana | live-watch | "never the source of a published number" (§8b). The exporter branches on only `l1`/`l3`/`perfspect`, silently dropping `l1_system` and `emon`, so its aggregates *will* disagree with SQL over this DB. |
| Langfuse | its own Postgres/ClickHouse/S3 | never merged; only `run_id` joins. SQLite is authoritative for tokens/cost — Langfuse's LiteLLM callback under-reports prompt tokens in multi-turn loops (measured 400,760 vs 22,904). |
| `harness/results/` | a colleague's 18+ GNR baselines | **never read or written by any phase** (HARD RULE) |

---

## 2. Entity model

```
                    ┌──────────────────────────────────┐
                    │ benchmarks        (mig 0004)     │
                    │  PK benchmark_id  (adapter slug) │
                    │  display_name, version, metadata │
                    └────────────────┬─────────────────┘
                                     │  runs.benchmark_id
                                     │  NOT a foreign key — deliberate:
                                     │  a run insert must never fail on a
                                     │  missing benchmark row. Plain index
                                     ▼  idx_runs_benchmark instead.
  ┌────────────────────────────────────────────────────────────────────────┐
  │ runs                                    schema.sql + mig 0004 + 0007   │
  │  PK run_id TEXT   (a handle, NOT chronological — ULIDs are PLANNED)    │
  │  start_time NOT NULL (epoch s) · end_time · status                     │
  │  provenance: hardware_sku · host_id · owner_id/owner_kind · model      │
  │              optimization_profile · numa_policy                        │
  │              agentsysperf_version (renamed by 0007) · result_digest    │
  │  metadata JSON  (everything outside the 15 promoted keys)              │
  └──┬─────────────┬──────────────┬──────────────┬──────────────┬──────────┘
     │ CASCADE     │ NO ACTION    │ CASCADE      │ CASCADE      │ CASCADE
     ▼             ▼              ▼              ▼              ▼
┌──────────────┐ ┌────────────┐ ┌─────────────┐ ┌───────────────┐ ┌──────────┐
│ task_results │ │ spans      │ │ measurements│ │analyzer_       │ │artifacts │
│  mig 0003    │ │ schema.sql │ │  mig 0002   │ │  verdicts      │ │ mig 0006 │
│ PK (run_id,  │ │ PK (run_id,│ │ PK meas._id │ │ mig 0005       │ │ PK art_id│
│     task_id) │ │    span_id)│ │ UNIQUE      │ │ UNIQUE (run_id,│ │ UNIQUE   │
│ logical_task_│ │ StepTrace  │ │ (run_id,    │ │  task_id,      │ │ (run_id, │
│  _key ◄ THE  │ │  v0.3      │ │  span_id,   │ │  analyzer_name)│ │  kind,   │
│  cross-run   │ │ tokens/cost│ │  layer)     │ │ verdict,       │ │  name)   │
│  agg key     │ │ /latency   │ │ 6 promoted  │ │ confidence,    │ │ path →   │
│ passed,      │ │ per step   │ │ cols + full │ │ evidence JSON, │ │ on-disk  │
│ duration_s   │ │            │ │ payload JSON│ │ recommendations│ │ file     │
└──────────────┘ └────────────┘ └─────────────┘ └────────────────┘ └──────────┘
     ▲                  ▲              ▲
     │ task_id          │ task_id      │ task_id      ← THREE DIFFERENT
     │ = adapter's      │ = parent_    │ = span_id      DERIVATION RULES
     │   real task id   │   span_id or │   .split('::') SHARING ONE NAME.
     │                  │   span_id    │   [-1]         Never join them. (§4)
     │                  │              │

  THE SWEEP TIER — sits ABOVE runs, additively:

  ┌────────────────────────────┐        ┌──────────────────────────────────┐
  │ sweeps        schema.sql   │  1:N   │ sweep_points        schema.sql   │
  │  PK sweep_id               │───────►│  PK (sweep_id, run_id)           │
  │  created_at NOT NULL       │NO      │  density = concurrency /         │
  │  vcpu_basis + _kind ◄ the  │ACTION  │    sweeps.vcpu_basis             │
  │    density DENOMINATOR     │        │  concurrency, replicate          │
  │  numa_policy, model,       │        │  throughput_per_min, elapsed_s   │
  │  replay_fixture, benchmark │        │  completed_trials,               │
  │  metadata.data_source ◄    │        │    p95_trial_latency_s           │
  │    'synthetic'|'measured'  │        │  node telemetry rolled up from   │
  └────────────────────────────┘        │    the l1_system layer           │
                                        │  run_id has NO FK to runs  ──────┼──┐
                                        └──────────────────────────────────┘  │
   Each density cell is ALSO an ordinary runs row (run_id                      │
   '{sweep_id}::d{density:g}_r{replicate}'), and the sweep_id itself   ────────┘
   is inserted as a PLACEHOLDER runs row so the sweep-level scaling
   verdict (filed under analyzer_verdicts.run_id = sweep_id) has an
   FK parent. A single sweep_id therefore appears as three different
   logical things in three tables. See §4 and §7.6.
```

Two structural points worth internalising before reading §3:

1. **`spans` is between `task_results` and `measurements` conceptually, not
   referentially.** There is no FK from `measurements.span_id` to `spans.span_id`.
   Measurements routinely exist for spans that were never written to `spans`: only
   the TB2/agent-loop path calls `store_spans`, so on the live store 9 of 10 runs
   have measurements and **zero** spans. An `INNER JOIN spans` silently drops every
   such run.
2. **Cascades are uneven.** `task_results`, `measurements`, `analyzer_verdicts` and
   `artifacts` cascade on run delete. `spans` is `NO ACTION` and `sweep_points` has
   no run FK at all, so `delete_run` prunes both in application code first —
   mandatory, not an optimisation: with `foreign_keys = ON`, deleting a run that
   still has spans would *fail the constraint*.

---

## 3. Table reference

### 3.1 `runs`

**Grain.** One row per benchmark run — *or* per sweep, *or* per sweep cell (see
gotchas). The top-level unit of work and the FK parent for everything else.

**Primary key.** `run_id TEXT PRIMARY KEY`. A TEXT PK is not a rowid alias, so
SQLite materializes `sqlite_autoindex_runs_1` and — importantly — does **not**
imply `NOT NULL` (`PRAGMA table_info` reports `notnull=0`).

| Column | Type | Meaning | Gotchas |
|---|---|---|---|
| `run_id` | TEXT PK | Caller-supplied run identifier; a handle, not a fingerprint | Free text in practice (`synthetic_cpu_1785191912`, `run-a1b2c3d4`). ULID run_ids are **PLANNED** (`STORAGE_UNIFICATION_PLAN.md` §1) so **`ORDER BY run_id` is NOT chronological** — always `ORDER BY start_time`. A single NULL is technically insertable. |
| `start_time` | INTEGER | Unix epoch **seconds** at run start; sort key for `list_runs`/`latest_run` | `NOT NULL`, no default. The writer coerces a missing key to `0`, so start-time-less runs sort last rather than erroring. Migrations 0003/0005 backfill placeholder rows with `0`. **0 means unknown/synthetic, not 1970.** |
| `end_time` | INTEGER | Unix epoch seconds at finish | NULL = still running *or* never reported. Not 0. The upsert uses `COALESCE(excluded.end_time, end_time)`, so a later write with `None` cannot erase it. |
| `hardware_sku` | TEXT, no default (since 0008) | CPU SKU string | **NULL means "not recorded" — never assume a SKU.** `schema.sql` originally shipped `DEFAULT 'Intel Xeon Platinum 8592+'`, the SKU of the authoring host, which silently mislabelled every run that did not supply one. Migration **0008** rebuilt `runs` to drop that default. The default was already dead in practice (`store_run_metadata` always binds an explicit `metadata.get('hardware_sku')`, i.e. NULL), so 0008 removes the mislabelling mechanism without retro-editing history: pre-0008 rows that inherited the literal are indistinguishable from rows genuinely on 8592+ and are preserved as-is. Real runs on hosts whose CPUID carries no marketing SKU record that raw placeholder string instead. |
| `model` | TEXT | LLM model name | Sweeps driven through the replay proxy write the literal `'agentsysperf-proxy'`, which means *the LLM was replayed, not called*. Do not aggregate beside real model names. |
| `total_tasks` | INTEGER, `DEFAULT 0` | Denormalized task count | Same dead-default trap (the writer binds `metadata.get('total_tasks')`, so the DDL `0` can never fire and an omitted key stores NULL). **Real callers do pass it**: `run_driver.py:344` sends `len(specs)`, and all 10 live runs are populated and agree with `task_results`. Still denormalized against `COUNT(*) FROM task_results` and **not enforced** — they can drift. `cli db ls` defends with `or 0`. |
| `passed_tasks` | INTEGER, `DEFAULT 0` | Denormalized pass count | Same: dead default, populated in practice by `run_driver.py`, unenforced. Prefer aggregating `task_results` for a guaranteed-consistent pass rate. |
| `metadata` | JSON | Every metadata key outside the 15 promoted ones | SQLite has no JSON type — a declared name with no matching affinity rules; it stores the TEXT written. Always `json.dumps(extra)`, so `'{}'` not NULL. **The upsert CLOBBERS it** (`metadata = excluded.metadata`, no merge) — a later partial write destroys the whole blob. Sweep provenance (`data_source`) lives in `sweeps.metadata`, not here. |
| `benchmark_id` | TEXT | Adapter slug (`synthetic-cpu`, `terminal-bench`) | **Deliberately a plain indexed column, NOT an FK** (0004's rationale: a run may be written before its benchmark row exists, and a run insert must never fail). Consequence: `benchmarks` can be empty while every run carries a slug — exactly what the live DB shows (two distinct slugs, neither registered). Note the slug normalisation: plugin names use underscores (`synthetic_cpu`), stored ids use hyphens. |
| `optimization_profile` | TEXT | `OptimizationProfile` applied, e.g. `amx_bf16` | NULL in all real runs. Free text, no enum. |
| `numa_policy` | TEXT | `unpinned` / `socket_pinned` / `interleaved` | Convention only, no CHECK. Duplicated on `sweeps.numa_policy`; a sweep and its cells can disagree. |
| `agentsysperf_version` | TEXT | Harness version — the provenance that makes a number defensible | **Renamed by migration 0007** from `agentperf_version` (which 0004 shipped). Pre-0007 DBs have the old name, and `persist_run` failed outright with *"table runs has no column named agentsysperf_version"*, rolling back the entire run write. NULL in all real runs — nothing populates it yet. |
| `result_digest` | TEXT | Content hash; tombstone identity for a withdrawn published run | NULL everywhere; nothing computes it. The soft-delete workflow it exists for is **PLANNED** (`STORAGE_UNIFICATION_PLAN.md` §3). |
| `owner_id` | TEXT | Attribution principal (`$USER` locally, an auth principal on a server) | `list_runs(owner_id=None)` returns **everyone's** runs (the shared view); passing a value scopes to one owner. `None` is not a filter-for-NULL. |
| `owner_kind` | TEXT | How `owner_id` was established: `os_user` / `oidc` / `token` | |
| `host_id` | TEXT | Hostname | **Cross-host comparisons MUST group on this** — `hardware_sku` alone does not distinguish two nodes of the same SKU. |
| `status` | TEXT | `running` / `complete` / `aborted` / `deleted` | Convention, no CHECK. NULL = unknown, **not** complete. `'deleted'` is the **PLANNED** soft-delete marker; the shipped `delete_run` is a hard delete, so it never appears. |

**Introduced by.** `schema.sql` (8 columns); columns 9–17 added by
`0004_benchmarks_and_provenance.sql` as nine additive `ALTER TABLE ADD COLUMN`;
`agentperf_version` → `agentsysperf_version` by `0007`.

**Writers.** `store_run_metadata` (17-column `INSERT … ON CONFLICT(run_id) DO
UPDATE` = 15 promoted keys + `run_id` + `metadata`, every field `COALESCE`-guarded
**except** `metadata`). Note `run_id` is **not** in the writer's `known_fields` set,
so a `run_id` key passed *inside* the metadata dict is not promoted — it stays in the
JSON blob and can then contradict the row's own primary key (verified:
`store_run_metadata(run_id='r1', metadata={'run_id': 'SHOULD_NOT_BE_PROMOTED'})`
stores that string in `metadata`). Also `persist_run`;
`sweep/harbor_sweep.py::_ensure_run_row` (placeholder rows for `sweep` and
`sweep_cell`); migrations 0003 and 0005 (`INSERT OR IGNORE` placeholders carrying
`metadata` `{"backfilled_orphan": true}`).

**Readers.** `get_run`, `list_runs`, `latest_run` (all `SELECT *`);
`store_task_result` (reads `benchmark_id` to build `logical_task_key`);
`delete_run`/`delete_benchmark`/`list_benchmarks`; `cli db ls|show`;
`markdown_report`, `xeon_pptx`, `demo_app`; DuckDB `ipc_by_benchmark`.

**Why it exists / what breaks without it.** It is the identity and provenance
anchor. Every other table hangs off `run_id` and four cascade-delete from it.
Without it there is no unit of publication — no way to say *which* machine, *which*
harness version, *which* profile produced a number, and no way to delete a bad
run's data. It is also why migrations 0003 and 0005 must backfill placeholder
`runs` rows for dangling children before adding their cascade FKs: no parent, no
migration.

---

### 3.2 `task_results`

**Grain.** One row per (run, task) — one task **attempt** within one run. *Not* one
row per logical task.

**Primary key and why.** Composite `PRIMARY KEY (run_id, task_id)`. The pre-0003
shape made `task_id` a **global** PK, so the same logical task run twice under two
`run_id`s collided and the second run's row silently overwrote the first. Cross-run
aggregation — "task X across N runs" — was structurally impossible, and so was any
measurement of run-to-run variance or regression. The composite key is the natural
identity: a task result only means anything relative to its run.

| Column | Type | Meaning | Gotchas |
|---|---|---|---|
| `run_id` | TEXT `NOT NULL` | Owning run | **Column order changed in 0003** — `run_id` is now column 0, `task_id` column 1 (baseline had `task_id` first). Any consumer doing `SELECT *` with positional indexing broke silently across this migration. Use named columns. |
| `task_id` | TEXT `NOT NULL` | Task identifier within the benchmark (`compile`, `linalg`, `ml_train`, …) | **Not globally unique** — that is the point of 0003. Also not the same namespace as `spans.task_id` or `measurements.task_id` (§4). |
| `logical_task_key` | TEXT | `'{benchmark_id}::{task_id}'` — the stable cross-run aggregation key | NULL in two cases: rows **copied by 0003** (no benchmark was knowable at migration time) and runs whose `benchmark_id` is NULL. A **snapshot** of `runs.benchmark_id` at write time — changing the run's slug later does not update it. `GROUP BY` on it silently drops NULL-key rows; repair with `COALESCE(logical_task_key, benchmark_id \|\| '::' \|\| task_id)`. |
| `workload_type` | TEXT | Task category; the indexed filter for `query_tasks(workload_type=…)` | **NULL on every real row** despite `idx_task_workload` existing — the runners never populate it. Filtering by it returns nothing. |
| `passed` | BOOLEAN `NOT NULL` | Verifier outcome | SQLite has no boolean type: NUMERIC affinity, stored as INTEGER 0/1 (`typeof(passed)='integer'`). The writer coerces `result.get('passed', False)`, so an omitted key means *false*. It is also the **only** field the upsert does not `COALESCE` — a re-store with a missing key **flips a passing task to failed**. |
| `duration_s` | REAL | Wall-clock duration in **seconds** | Seconds here, while `measurements.duration_us` and every `spans.*_us` are microseconds. Cross-table math needs 1e6. NULL ≠ 0. |
| `num_turns` | INTEGER | Agent turns (LLM round-trips) | NULL on all real rows. The write path *does* pass it from the adapter's `extra`, so NULL means the adapter didn't report it. |
| `num_commands` | INTEGER | Shell commands executed | NULL on all real rows. The TB adapter **does** report it (`benchmarks/terminal_bench/adapter.py:376-377` populates `extra['num_commands']`), but `run_driver.py:322` forwards only `num_turns` into the task-result dict, so it is dropped in the driver — a one-line gap, not a missing measurement. |

**Constraints.** `FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE`
(added by 0003; the baseline FK had no cascade), enforced only because the store
sets `PRAGMA foreign_keys = ON`. `sqlite_autoindex_task_results_1` UNIQUE on the
composite PK. Indexes `idx_task_logical(logical_task_key)`,
`idx_task_workload(workload_type)`.

**Introduced by.** `schema.sql`; **completely rebuilt** by
`0003_task_results_composite_pk.sql`.

**Writers.** `store_task_result` — first runs `SELECT benchmark_id FROM runs WHERE
run_id = ?` to derive `logical_task_key`, so **the `runs` row must already exist**
or the key is silently NULL (which is why `persist_run` stores metadata first);
then an 8-column upsert on `(run_id, task_id)`. Reached via
`persist_run(task_results=[(task_id, dict), …])` from `run_driver.py`.

**Readers.** `query_tasks(run_id, workload_type=None)`, `cli db show`,
`markdown_report.py`, `xeon_pptx.py`.

**Why it exists.** The pass/fail plus latency system of record — the numbers that
get published. `logical_task_key` is what turns "task X's p95 across all runs of
benchmark B" into a one-line `GROUP BY` instead of a string-parsing exercise.
SQLite cannot `ALTER` a primary key, hence 0003's documented rebuild (create
`_new` → copy → drop → rename), with orphan-run backfill first so the new cascade
FK loses no rows.

---

### 3.3 `measurements`

**Grain.** One row per (run_id, span_id, layer) — one probe emission for one span
from one probe layer. **One span typically yields several rows**: in the live DB the
span `compile` has an `l1` row, an `l1_system` row and an `l3` row.

**Primary key and why.** `measurement_id INTEGER PRIMARY KEY AUTOINCREMENT` is a
surrogate; the real identity is `UNIQUE(run_id, span_id, layer)`, which is what the
upsert targets. The surrogate exists so an AUTOINCREMENT rowid is available and the
natural key can evolve. 0002's comment records that the uniqueness was **verified
distinct across all real data before shipping** (l1/l3 emit one record per
span+layer; perfspect uses per-task system scope with distinct span_ids), so the
upsert can never silently collapse two genuine records.

| Column | Type | Meaning | Gotchas |
|---|---|---|---|
| `measurement_id` | INTEGER PK AUTOINCR | Surrogate row id | **Internal.** Monotonic, never reused, tracked in `sqlite_sequence`. Do not persist as an external reference — a re-import assigns new ones. |
| `run_id` | TEXT `NOT NULL` | Owning run | FK CASCADE to `runs`; the parent must pre-exist. |
| `span_id` | TEXT `NOT NULL` | The span this record annotates | **No FK to `spans`.** Measurements exist for spans never written to `spans` — on the live DB only the one TB2 run has any `spans` rows at all, so an `INNER JOIN spans` drops every other run's measurements. Do not `INNER JOIN spans`. |
| `layer` | TEXT `NOT NULL` | Which probe produced the row | Shipped values: `l1` (per-process resource), `l1_system` (whole node), `l3` (perf counters), `emon`, `perfspect`. **The DDL comment omits `l1_system`** even though it is a third of real rows and is `ScalingAnalyzer.input_layers`. Free text, no CHECK. **Layer determines which promoted columns are non-NULL — always filter by layer before averaging one.** |
| `task_id` | TEXT | Denormalized task attribution | **Derived**: `span_id.split('::')[-1] if '::' in span_id else span_id`. Deliberately verbatim the Prometheus exporter's label rule, so a store-backed exporter emits the same label set. **Not** guaranteed equal to `task_results.task_id`, and a different rule from `spans.task_id`. |
| `duration_us` | INTEGER | Span duration in **microseconds** | Populated for `l1` only; NULL for `l1_system`/`l3`, which emit `duration_s` (seconds) inside the payload instead. The plan doc proposed REAL; shipped INTEGER. |
| `cpu_time_s` | REAL | CPU time consumed, **seconds** | **The canonical NULL-vs-zero trap.** NULL on every non-`l1` row (the probe does not emit the key); within `l1`, both exact `0.0` and real positive values are common — measured 11 zeros vs 70 positive. `0.0` = the probe measured and got zero. `AVG()` ignores the NULLs but *includes* the zeros, dragging the mean down. |
| `cpu_pct_mean` | REAL | Mean CPU utilization **percent** over the span | `l1` only. `l1_system`'s node-level equivalents live **only** in the payload as `cpu_avg`/`cpu_p50`/`cpu_p95`/`cpu_peak` and are not promoted — so "average CPU" means a different thing per layer. |
| `rss_kb_peak` | INTEGER | Peak RSS in **kilobytes** | KB, not bytes, not MB (contrast `sweep_points.mem_avail_mb_min`, which is MB). `l1` only. |
| `ipc` | REAL | Instructions per cycle | `l3` and `perfspect` only; NULL on every `l1`/`l1_system`/`emon` row. `duckdb_analytics.ipc_by_benchmark` correctly guards `WHERE m.ipc IS NOT NULL` — but that guard alone is **not enough**: it admits both layers, and perf-stat and perfspect disagree by ~2× on the same workload (measured `synthetic-cpu`: `l3` n=22 avg 2.373 vs `perfspect` n=12 avg 1.024). **Add `AND layer = …` and report per layer.** Its sibling `avg(cache_miss_pct)` column is silently `l3`-only inside that same ipc-gated result set. |
| `cache_miss_pct` | REAL | Cache miss rate, **percent** | **`l3` only in practice.** `ipc` is promoted from both `l3` and `perfspect`, but perfspect payloads carry no `cache_miss_pct` key at all (`perfspect/measurement.py` has no mapping for it; only `l3_perf/probe.py` emits it), so it is NULL on every perfspect row — measured `l3` 64/64 non-NULL, `perfspect` 0/13. Which cache level is *not* encoded in the name; the `l3` payload separately carries `llc_miss_per_s` and `branch_miss_pct`. |
| `payload` | JSON `NOT NULL` | The full probe payload — **the source of truth** | Always `json.dumps` of a dict. Key sets differ wildly by layer (see below). Byte-equal round-trip is the *intended* contract, but the test that pins it (`test_measurements_roundtrip.py::test_p1_roundtrip_real_docs_payloads`) is `skipif`-gated on `docs/*/measurement_records.json`, which does **not** exist in the repo — so `pytest -rs` reports it SKIPPED and byte-equality is **not** actually enforced in CI today. `node_id` and `kind` live **only** here (the plan proposed them as columns; they did not ship). |
| `seq` | INTEGER | 0-based insertion order within the run | `query_measurements ORDER BY seq` is the **only** stable ordering — `measurement_id` order can diverge after upserts. The counter continues past the run's existing max, so **re-storing the same records assigns new higher `seq` values** (relative order preserved, absolute values not stable, gaps appear). Load-bearing for the PhaseProfiler, which parses turn order from `span_id` in `seq` order. |
| `schema_version` | TEXT, `DEFAULT '0.4'` | Measurement-row schema version | **The one default that actually fires** — `store_measurements` omits the column from its 12-column INSERT, so all real rows read `'0.4'`. Unrelated to `spans.schema_version` (`'0.3'`) and to `PRAGMA user_version` (7). |

**Per-layer payload keys (measured).**

| Layer | Payload keys |
|---|---|
| `l1` | 9 keys: `cpu_pct_mean, cpu_pct_peak, cpu_time_s, duration_us, kind, node_id, num_threads_peak, rss_kb_peak, sample_count` — plus an optional 10th, `phase`, on the six-phase adapter path (measured on 51 of 81 live `l1` rows) |
| `l1_system` | 15 keys: `cpu_avg, cpu_p50, cpu_p95, cpu_peak, ctx_sw_per_s_avg, duration_s, iowait_pct_avg, kind, logical_cpus, mem_avail_mb_min, mem_used_mb_max, node_id, runqueue_avg, runqueue_max, sample_count` |
| `l3` | 9 keys: `branch_miss_pct, cache_miss_pct, duration_s, events, ipc, kind, llc_miss_per_s, node_id, sample_counts` |
| `perfspect` | **53 keys (measured** — identical on all 13 live rows). Wide but flat: `ipc`, `cpi`, `kernel_cpi`, frequency, power, per-instruction TLB/cache MPIs, LLC and NUMA bandwidth, IO/DDIO breakdowns. **No TMA keys.** The TMA hierarchy *is* computed in `perfspect/measurement.py:466-489`, but it is gated on `frontend_bound`/`backend_bound`/`bad_speculation`/`retiring` all being present, and perfspect emitted no `TMA_*(%)` CSV columns on this rig — so `tma_dominant_bucket`/`tma_classification` reach **no** stored payload (`[k for k in keys if 'tma' in k]` → `[]`). |
| `emon` | **3 keys: `arch`, `csv_path`, `mode`** — a **POINTER to an off-DB CSV, with no metrics at all** (e.g. `{"csv_path": ".../emon_run_system_view_details.csv", "mode": "whole_run", "arch": ""}`). `emon` rows also populate **zero** promoted columns (verified: `duration_us`, `cpu_time_s`, `cpu_pct_mean`, `rss_kb_peak`, `ipc`, `cache_miss_pct` all 0 non-NULL across every `emon` row). **Any layer-agnostic metric aggregation must exclude `emon`** — it contributes rows and no numbers. |

Note `l1` emits `duration_us` (promoted) while `l1_system`/`l3` emit `duration_s`
(not promoted) — the same conceptual field is microseconds-in-a-column or
seconds-in-JSON depending on layer.

**Indexes.** `idx_meas_run_layer(run_id, layer)` — the commonest access pattern;
`idx_meas_span(run_id, span_id)` — per-span drilldown correlating one span's l1 +
l3 + perfspect rows; `idx_meas_run_task(run_id, task_id)` — per-task rollup
matching the exporter's label set; plus the UNIQUE autoindex, which also serves
bare `WHERE run_id = ?` by left prefix.

**Introduced by.** `0002_measurements.sql` — **not** in `schema.sql`. A fresh DB
gets this table only because the migration runner applies 0002 on top of the
baseline.

**Writers.** `store_measurements` (12-column upsert on the natural key overwriting
every non-key column including `seq`); `persist_run(records=…)`; `RunContext.stop()`
dual-writes `ctx._records` here after writing `measurement_records.json`.

**Readers.** `query_measurements(run_id, layer=/span_id=/task_id=)` returns
`span_id, layer, task_id, payload ORDER BY seq` — **it does not return the six
promoted columns or `seq`**; those are reachable only via raw SQL or DuckDB. Also
`dashboard_data.get_records`, `cli db show`, `markdown_report`, and DuckDB
`measurement_count`/`ipc_by_benchmark`/`export_parquet`.

**Why it exists.** This table **ended the split-brain**. Before 0002,
`store_measurements` was a no-op and every L1/L3/EMON/perfspect record lived only
in per-directory JSON, so the DB was not the system of record for hardware data.
The promote-six-hot-keys-plus-keep-the-JSON-tail design is the whole storage
philosophy in one table: dashboards get typed indexed columns for what they plot,
perfspect's ~53 keys need zero migration, and the promote-when-hot lifecycle means
you migrate to optimize a metric, never to add one. Without it: no cross-run
hardware analysis, no store-backed Prometheus exporter, no DuckDB acceleration.

---

### 3.4 `spans`

**Grain.** One row per recorded agent execution **step** — an LLM call, a tool call,
a routing decision, a plan-build step, or an agent-step boundary. Mirrors
`src/trace/schema.py::StepTrace` v0.3.

**Primary key.** Composite `PRIMARY KEY (run_id, span_id)` — already correct in the
baseline and untouched by every migration. `span_id` is unique only **within** a run.

| Column | Type | Meaning | Gotchas |
|---|---|---|---|
| `span_id` | TEXT `NOT NULL` | Step identifier within the run | `PRAGMA` reports `pk=2` while `run_id` is `pk=1`, even though `span_id` is physically column 0. `query_spans ORDER BY span_id` is documented as "turn order" but is a **lexicographic** sort, so `turn_10` precedes `turn_2`. |
| `run_id` | TEXT `NOT NULL` | Owning run | |
| `parent_span_id` | TEXT | Outer invocation span (typically an `agent_step`) | NULL marks a root span, and is the input to the `task_id` derivation below. |
| `task_id` | TEXT | Task attribution; the indexed filter for `query_spans(task_id=…)` | **Computed by the writer**, not supplied: `parent_span_id or span_id`. This is a **third** `task_id` rule. Do not assume it joins to `task_results.task_id`. |
| `span_kind` | TEXT `NOT NULL` | Step type | Closed enum in `trace/schema.py::SpanKind`: `llm_call`, `tool_call`, `route_decision`, `plan_build`, `agent_step`. Stored as bare TEXT with no CHECK; the writer unwraps the enum via `getattr(kind,'value',kind)`. The DDL comment omits `route_decision`/`plan_build`. The enum is **locked at 0.3** — adding a kind requires a schema bump, because analyzers, reporters and exporters dispatch on it. |
| `node_id` | TEXT | Agent-graph node name, or the emitter's synthesized invocation id | `StepTrace` defaults it to `''`, not `None` — expect empty string, not NULL. |
| `start_ts_us` | INTEGER `DEFAULT 0` | Start time, **microseconds** since epoch | µs (not ms) was chosen so Perfetto consumes the trace natively and it converts cleanly to OpenTelemetry. Legacy rows carry 0 — treat 0 as unknown. |
| `end_ts_us` | INTEGER `DEFAULT 0` | End time, microseconds since epoch | Same. |
| `duration_us` | INTEGER `DEFAULT 0` | Step duration, microseconds | Default 0 at **both** DDL and StepTrace layers, so an unmeasured step is indistinguishable from a zero-duration one. Unlike `measurements.duration_us` there is no NULL to signal "not measured". |
| `model_id` | TEXT | Model used; populated when `span_kind = 'llm_call'` | |
| `tokens_in` | INTEGER `DEFAULT 0` | Prompt tokens | Defaults 0 at both layers — a `tool_call` legitimately reads 0, and so does an `llm_call` whose counts were never captured. `SUM(tokens_in)` over all spans is meaningless without `WHERE span_kind='llm_call'`. |
| `tokens_out` | INTEGER `DEFAULT 0` | Completion tokens | Same. |
| `cost_usd` | REAL `DEFAULT 0.0` | Dollar cost | `0.0` for non-LLM spans and uncosted LLM spans alike. Never NULL, so "cost unknown" is undetectable. |
| `tool_name` | TEXT | Tool invoked; populated when `span_kind = 'tool_call'` | |
| `resource_tier` | TEXT | Resource class the tool call was dispatched to | |
| `status` | TEXT `DEFAULT 'ok'` | Step outcome | Defaults to `'ok'` at both layers, so an **unset status reads as success**. Detect failure via `status <> 'ok' OR error IS NOT NULL`. The upsert overwrites `status`, so a re-store can flip a failure to ok. |
| `error` | TEXT | Error message | |
| `extra` | JSON | Free-form provenance; per-step hardware rollups get stashed here | Always `json.dumps(dict(extra or {}))` → `'{}'`, not NULL. **Does not** capture the dropped StepTrace fields below — they are simply lost. |
| `schema_version` | TEXT `DEFAULT '0.3'` | StepTrace schema version | Written on INSERT but **omitted from the `ON CONFLICT DO UPDATE SET` list** — re-storing a span with a newer version silently keeps the old tag while updating every other field, producing a row whose version lies about its contents. |

**Constraints.** `FOREIGN KEY (run_id) REFERENCES runs(run_id)` with **`NO ACTION`**
on delete (confirmed by `PRAGMA foreign_key_list`). This is why `delete_run` must
issue `DELETE FROM spans WHERE run_id = ?` **first**: with `foreign_keys = ON` the
run delete would otherwise *fail the constraint*, not orphan silently. Indexes
`idx_spans_task(run_id, task_id)` and `idx_spans_kind(run_id, span_kind)` — the
latter serves the token/cost aggregation path that must exclude `tool_call` rows.

**Introduced by.** `schema.sql`. Never altered by any migration.

**Writers.** `store_spans` (19-column upsert on `(run_id, span_id)`, updating 16 of
the 17 non-key columns — it omits `schema_version`, see §8.1);
`persist_run(spans=…)`; `RunContext.stop()` dual-writes
`ctx._step_traces`.

**LOSSY.** `store_spans` **silently drops 12 StepTrace fields** with no column and
no folding into `extra`: `runtime_id`, `session_id`, `deployment_id`,
`routing_reason`, `quality_score`, `kv_cache_hit`, `dispatch_mode`, `wall_clock_ms`,
`queue_wait_ms`, `worker_id`, `batch_id`, `batch_size`. Both sides claim
`schema_version` `'0.3'` and the module docstring calls the schema byte-compatible
with AgentOptimizer's `agentflow/core/schema.py` v0.3 — but a StepTrace
round-tripped through this store loses over a third of its fields, including the
routing and caching provenance.

**Readers.** `query_spans(run_id, task_id=None, span_kind=None)` → `SELECT * ORDER
BY span_id`; `xeon_pptx.py` accesses it defensively via
`getattr(store,'query_spans',None)` plus a `start_ts_us` filter.

**Why it exists.** The step-level counterpart to the hardware measurements: per-step
tokens, cost, latency and tool exit status. Without it you can say "this run took
40 s at 60% CPU" but not "the LLM calls were 30 s of that and cost $0.12" — the
decomposition that makes the CPU-architecture story actionable. It is also the
interop surface: the µs timestamps make it directly Perfetto/OTel-consumable.

---

### 3.5 `analyzer_verdicts`

**Grain.** One row per (run_id, task_id, analyzer_name) — one analyzer's conclusion
about one task within one run.

**Primary key and why.** `verdict_id INTEGER PRIMARY KEY AUTOINCREMENT` is a
surrogate; the real identity is `UNIQUE(run_id, task_id, analyzer_name)` added by
0005. The surrogate is retained because 0005's dedup keeps `MAX(verdict_id)` (most
recent) per identity, and `query_verdicts` orders by it.

| Column | Type | Meaning | Gotchas |
|---|---|---|---|
| `verdict_id` | INTEGER PK AUTOINCR | Surrogate id; also the recency proxy | **Internal.** `ORDER BY verdict_id` is insertion order, not analyzer or task order. Not stable across a re-import. |
| `run_id` | TEXT `NOT NULL` | Owning run | **Overloaded**: for scaling analysis the sweep runner stores the verdict under `run_id = sweep_id`, not a real benchmark run. That is why `_ensure_run_row` inserts a placeholder `runs` row for the sweep_id, and why 0005 had to backfill placeholders for legacy dangling verdicts before adding its cascade FK. **A `run_id` here is not necessarily a benchmark run.** |
| `task_id` | TEXT `NOT NULL` | Which task the verdict is about | Populated from `AnalysisResult.span_id` — **a span id in a column named `task_id`** — with the sentinel string `'unknown'` substituted when falsy. Because it is part of the UNIQUE key, **all run-level verdicts from one analyzer collide on `(run_id,'unknown',analyzer_name)` and upsert over each other; only the last survives.** |
| `analyzer_name` | TEXT `NOT NULL` | Which analyzer produced it | Measured on the live store (7 values): `breakdown`, `cache`, `cpu_bound`, `emon`, `memory_bandwidth`, `memory_leak`, `phase_profiler` — plus `scaling` for sweeps. Free text from `AnalysisResult.analyzer_name`, which **defaults to `''`** in the dataclass — an analyzer that forgets to set it writes an empty name that still satisfies NOT NULL and participates in the UNIQUE key. |
| `verdict` | TEXT `NOT NULL` | The conclusion label | Measured on the live store (16 values): `core_bound`, `dram_bound`, `emon_ddio_effectiveness`, `emon_tma_frontend_bound`, `gc_pressure`, `growing`, `io_bound`, `l3_pressure`, `l3_resident`, `leak_suspected`, `memory_bottleneck_moderate`, `memory_bottleneck_severe`, `memory_bound`, `no_memory_bottleneck`, `orchestration_heavy`, `phase_profile_reason`. **Analyzer-specific vocabulary with no shared enum and no CHECK — switch on `analyzer_name` first.** |
| `confidence` | REAL `NOT NULL` | 0.0–1.0 | No default, no range CHECK — nothing prevents 1.7. |
| `evidence` | JSON `NOT NULL` | The supporting numbers | Always `json.dumps(dict(...))` → `'{}'` when empty. `query_verdicts` calls `json.loads` **unguarded**, and the surrounding `except sqlite3.Error` does not catch `json.JSONDecodeError` — a malformed blob escapes to the caller. Shape is analyzer-specific and unversioned. |
| `recommendations` | JSON | Ordered list of actionable strings | A JSON **array**, not an object — the only one in the schema. `query_verdicts` maps falsy → `[]`, so NULL and `'[]'` are indistinguishable through the API. |

Both vocabularies grow with every new analyzer plugin, so **regenerate rather than
re-guess** — a dispatch built from a stale list silently falls through:

```sql
SELECT analyzer_name, verdict, COUNT(*) FROM analyzer_verdicts
GROUP BY analyzer_name, verdict ORDER BY analyzer_name, verdict;
```

**Constraints.** `FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE`
(added by 0005; baseline had no cascade); `UNIQUE(run_id, task_id, analyzer_name)`;
index `idx_verdict_analyzer(analyzer_name, verdict)` — note it does **not** lead
with `run_id`, so `query_verdicts(run_id, analyzer_name=…)` relies on the UNIQUE
autoindex instead.

**Introduced by.** `schema.sql`; **rebuilt** by `0005_verdict_unique.sql`.

**Writers.** `store_analysis_results` (7-column upsert); `persist_run(verdicts=…)`;
`run_driver.py`; `harbor_sweep.py`, which stores the scaling verdict under
`run_id = sweep_id` using `dataclasses.replace` to put the grouping key in the
`span_id` slot.

**Readers.** `query_verdicts(run_id, analyzer_name=None)` → `verdict_id, task_id,
analyzer_name, verdict, confidence, evidence, recommendations ORDER BY verdict_id`
(**`run_id` is not returned**); `xeon_pptx.py` per-analyzer slides;
`markdown_report.py`; `live_dashboard.py`; `demo_app.py`.

**Why it exists.** Persisted **analysis**, so the interpretation layer does not
re-derive itself from raw counters on every report render, and so a published claim
("this workload is orchestration-heavy") is auditable with its evidence and
confidence attached. Before 0005, `store_analysis_results` did a plain INSERT, so
re-running analyzers or re-importing a run **multiplied** verdict rows and every
dashboard count was inflated. The UNIQUE key made the write idempotent (pinned by
`test_p3_schema.py::test_verdict_upsert_not_duplicated`); 0005's rebuild dedups on
copy by keeping `MAX(verdict_id)` per identity.

---

### 3.6 `benchmarks`

**Grain.** One row per benchmark suite (adapter). The top-level entity runs belong to.

**Primary key and why.** `benchmark_id TEXT PRIMARY KEY` — the adapter **slug**,
deliberately human-meaningful and stable rather than surrogate, because it is
embedded as a string in `runs.benchmark_id` and inside
`task_results.logical_task_key`.

| Column | Type | Meaning | Gotchas |
|---|---|---|---|
| `benchmark_id` | TEXT PK | Adapter slug matching the registered adapter name | TEXT PK → not a rowid alias → `notnull=0`, so NULL is technically insertable. |
| `display_name` | TEXT | Human label, e.g. `Terminal-Bench` | `COALESCE`-guarded in the upsert, so a later write cannot blank it. NULL for benchmarks that exist only as a `runs.benchmark_id` value (the UNION arm of `list_benchmarks`). |
| `version` | TEXT | Suite version — provenance for "which task set produced this score" | `COALESCE`-guarded. **Runs do not record which benchmark *version* they ran**, only the slug, so a suite version bump is invisible in the run record. |
| `created_at` | INTEGER | Unix epoch **seconds** of registration | Writer coerces a missing key to `0`, not NULL. Not in the `DO UPDATE` list at all, so a re-registration cannot change it. |
| `metadata` | JSON | Keys other than the three above | `json.dumps(extra)` → `'{}'`. Upsert **clobbers**, no merge. `list_benchmarks` does not decode it (it isn't selected). |

**Introduced by.** `0004_benchmarks_and_provenance.sql`.

**Writers.** `store_benchmark` (5-column upsert). Note it **commits directly** via
`conn.commit()` rather than `self._commit()`, so it would break `persist_run`
atomicity if it were ever called inside one. `delete_benchmark` removes the row
after app-level-deleting all its runs.

**Readers.** `list_benchmarks` only — registered rows `LEFT JOIN runs` for a run
count, **UNIONed** with `DISTINCT runs.benchmark_id` for unregistered slugs.

**Why it exists.** Without it there is nowhere to record what a benchmark *is*
(display name, suite version) independent of any single run, and the UI has no
canonical list to filter by. 0004 deliberately made `runs.benchmark_id` a plain
indexed column rather than an FK so a run insert can never fail on a missing
benchmark row — which makes this table genuinely **optional metadata enrichment**,
and is why `list_benchmarks` must UNION in the run-derived slugs. The cost of that
choice: `delete_benchmark` cannot rely on a cascade and instead loops `delete_run`.

**Empirical proof of the design:** `benchmarks` has **0 rows** while every real run
carries a slug — the live store has two (`synthetic-cpu`, `terminal-bench`), neither
registered, so `v_benchmark_summary` returns 2 rows and both report
`is_registered = 0`. Consumers **must** use `list_benchmarks` (or the UNION pattern
in `v_benchmark_summary`), never `SELECT FROM benchmarks`.

---

### 3.7 `artifacts`

**Grain.** One row per registered on-disk file belonging to a run — a **pointer**,
never a blob.

**Primary key and why.** `artifact_id INTEGER PRIMARY KEY AUTOINCREMENT`; natural
identity is `UNIQUE(run_id, kind, name)`. The surrogate also encodes recency, which
`get_artifact_path` exploits (`ORDER BY artifact_id DESC`).

| Column | Type | Meaning | Gotchas |
|---|---|---|---|
| `artifact_id` | INTEGER PK AUTOINCR | Surrogate id, doubles as recency ordering | **Internal.** `get_artifact_path` returns the most recent match (DESC) while `list_artifacts` returns ascending — the two methods disagree on ordering. |
| `run_id` | TEXT `NOT NULL` | Owning run — the addressing key | Artifacts are looked up **by run_id**, never by globbing `/tmp`. |
| `kind` | TEXT `NOT NULL` | Artifact category | DDL documents `emon_csv \| scaling_plot \| scaling_results \| analysis_report \| …`. Code actually **writes** `emon_csv` and the undocumented `measurement_records`; code actually **reads** `emon_csv`. Nothing writes `scaling_plot`/`scaling_results`/`analysis_report` — `demo_app` still glob-discovers those. Free text, no CHECK. |
| `name` | TEXT `NOT NULL` | File basename, or a logical name | Part of the UNIQUE key, so two different files sharing a basename under the same `(run_id, kind)` collide and the second overwrites the first's path. |
| `path` | TEXT `NOT NULL` | Resolvable filesystem path | Stored as given (absolute for everything registered today). **Nothing validates existence at write time**; `get_artifact_path` filters by `p.exists()` and returns `None` if nothing is on disk — so a valid registration silently becomes invisible after the file is deleted or the DB is read on another host. |
| `created_at` | INTEGER | Registration time | **Hardcoded `0`** in `store_artifact`'s VALUES list. Always 0 for every artifact the shipped API writes. Useless as a timestamp; use `artifact_id` for recency. |
| `metadata` | JSON | Optional descriptors | **Hardcoded NULL**, with no parameter to set it. The only JSON column that is genuinely NULL rather than `'{}'`. Not updated by the upsert either (only `path` is). |

**Constraints.** `FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE`
— deleting a run removes artifact **rows** but never touches the files on disk.
`UNIQUE(run_id, kind, name)`; index `idx_artifacts_run_kind(run_id, kind)`.

**Introduced by.** `0006_artifacts.sql`.

**Writers.** `store_artifact(run_id=, kind=, name=, path=)` — uses `self._commit()`,
so it *is* `persist_run`-safe. Called only from the EMON collection drivers, which
ship with the separate Intel-only `agentsysperf-emon` plugin.

**Readers.** `get_artifact_path(run_id, kind=None, name=None)` → first **existing**
path, `ORDER BY artifact_id DESC`; `list_artifacts(run_id, kind=None)` →
`run_id, kind, name, path ORDER BY artifact_id` (ascending, and **without** an
existence filter, so it can list paths that no longer resolve). Wrapped by
`dashboard_data.get_artifact_path` (store first, then a legacy dir+glob fallback);
consumed by `demo_app.py` and the EMON analyzer (the `agentsysperf-emon` plugin),
both of which need a real on-disk file.

**Why it exists.** Bulk per-run files — EMON pyEDP CSVs, scaling PNGs,
`all_results.json`, `analysis_report.json` — do not belong in a relational store as
blobs, but they **do** need to be addressable by `run_id` instead of rediscovered by
globbing scattered `/tmp` directories (the pre-0006 status quo, which broke whenever
a scratch dir was cleaned). This table makes "give me run X's EMON CSV" a query.
Without it, `EmonAnalyzer` and image rendering fall back to fragile glob heuristics
— which is still the live fallback path in `dashboard_data.get_artifact_path`.

---

### 3.8 `sweeps`

**Grain.** One row per concurrency-scaling **sweep**. A sweep groups many runs (one
per density point × replicate) and sits **above** `runs`.

**Primary key.** `sweep_id TEXT PRIMARY KEY` (observed convention
`f'sweep_{int(time.time())}'`).

| Column | Type | Meaning | Gotchas |
|---|---|---|---|
| `sweep_id` | TEXT PK | Sweep identifier | TEXT PK → `notnull=0`. **Critical overload:** the same string is also inserted as a `runs.run_id` placeholder so the scaling verdict (filed under `analyzer_verdicts.run_id = sweep_id`) has an FK parent. A sweep_id appears in **three tables as three different logical things**. |
| `created_at` | INTEGER `NOT NULL` | Unix epoch seconds; sort key for `query_sweeps` | No default; the writer coerces a missing key to `0`, so such a sweep sorts last rather than erroring. |
| `hardware_sku` | TEXT | CPU SKU of the sweep host | Duplicated on every cell's `runs.hardware_sku` with no consistency enforcement. Unlike `runs.hardware_sku` there is **no DDL default** here. |
| `vcpu_basis` | INTEGER | The **denominator** for density — vCPU count (physical cores on EMR) | **The most important column for not misreading a sweep.** `density = concurrency / vcpu_basis`, so comparing density across sweeps with different `vcpu_basis` (or different `vcpu_basis_kind`) compares different things. |
| `vcpu_basis_kind` | TEXT | What `vcpu_basis` counts: `physical_cores` / `logical_cpus` | Disambiguates a 2× SMT factor. Convention only. **A NULL here makes every density in the sweep uninterpretable.** |
| `numa_policy` | TEXT | `unpinned` / `socket_pinned` / `interleaved` | Duplicated on `runs.numa_policy`, no enforcement that they agree. **Stored but never applied** by the sweep runner (§8), so `socket_pinned` and `unpinned` sweeps ran byte-identical. |
| `model` | TEXT | Model used | `'agentsysperf-proxy'` means the LLM was **replayed**, not called. |
| `replay_fixture` | TEXT | Path to the replay fixture — which recorded trajectory | NULL means live-LLM mode (or a dry run). A host-local path that will not resolve elsewhere. |
| `benchmark` | TEXT | The benchmark swept | **Named `benchmark`, not `benchmark_id`** — inconsistent with `runs.benchmark_id` and `benchmarks.benchmark_id`, and there is no FK. Nothing joins sweeps to benchmarks. |
| `metadata` | JSON | Keys outside the eight promoted ones | **Carries the integrity flag**: `data_source = 'synthetic'` (dry-run, modeled points, no Harbor) or `'measured'`, precisely so a dashboard can badge a synthetic sweep. **A consumer that ignores `metadata.data_source` can publish modeled numbers as measured ones.** Clobbered, not merged, on upsert. |

**Introduced by.** `schema.sql`. Never altered by any migration.

**Writers.** `store_sweep_metadata` (10-column upsert, all `COALESCE`-guarded except
`metadata`), called from `sweep/harbor_sweep.py`. **Commits directly** via
`conn.commit()`, not `self._commit()`.

**Readers.** `query_sweeps()` → `SELECT * ORDER BY created_at DESC`, metadata
decoded. Consumed by `demo_app.py` (Concurrency Sweep page) and `live_dashboard.py`.

**Why it exists.** A scaling analysis spans **multiple runs**, which the per-run
analyzer model structurally cannot express — so sweeps sit above runs, **additively**
(each density cell remains an ordinary `runs` row). This table carries the *shared
interpretation context*: without `vcpu_basis` + `vcpu_basis_kind` the density axis
is a bare ratio with no denominator; without `numa_policy`/`replay_fixture` the knee
is not reproducible; without `metadata.data_source` you cannot tell a modeled
dry-run curve from a measured one.

---

### 3.9 `sweep_points`

**Grain.** One row per **density cell** of a sweep = one (density, concurrency,
replicate) operating point, 1:1 with the per-cell run that produced it. This is the
cell-level rollup the `ScalingAnalyzer` reads.

**Primary key and why.** Composite `PRIMARY KEY (sweep_id, run_id)` — identity is
"this cell of this sweep". `run_id` is in the key rather than `replicate` because
each cell **is** a distinct run (observed convention
`'{sweep_id}::d{density:g}_r{replicate}'`).

| Column | Type | Meaning | Gotchas |
|---|---|---|---|
| `sweep_id` | TEXT `NOT NULL` | Owning sweep | The only FK-constrained column in this table. |
| `run_id` | TEXT `NOT NULL` | The per-cell run this point summarizes | **NO foreign key to `runs`** — confirmed by `PRAGMA foreign_key_list`, which shows only the sweeps FK. A sweep_point can reference a nonexistent run and nothing complains; `harbor_sweep` compensates by calling `_ensure_run_row` first. It is also why `delete_run` must app-level `DELETE FROM sweep_points WHERE run_id = ?`. The writer passes `point.get('run_id')`, so a point dict missing the key raises an IntegrityError that is **caught and logged, silently dropping the cell**. |
| `density` | REAL | `concurrency / vcpu_basis` — agents per vCPU. The sweep's X axis | **Meaningless without `sweeps.vcpu_basis` AND `vcpu_basis_kind`.** Derived and denormalized. Half of `idx_sweep_points_density` and the `ORDER BY` of `query_sweep_points`, so a NULL scrambles the curve. |
| `concurrency` | INTEGER | N agents run in parallel for this cell | **The requested ceiling, not achieved parallelism.** Harbor's `-n` is an `asyncio.Semaphore` ceiling over `n_attempts × n_tasks`, so peak concurrency is `min(n, n_trials)`; on a 10-task/1-attempt set every cell from d=0.25 to d=3.0 actually ran ~10 concurrent agents. |
| `replicate` | INTEGER | Replicate index for this (density, concurrency) point | **Not part of the PK** — replicates are distinguished only by having their own `run_id`. Aggregating a density point means `GROUP BY density` and averaging across replicates. |
| `elapsed_s` | REAL | Cell wall time, **seconds** | |
| `throughput_per_min` | REAL | `trials / (elapsed_s / 60)` — trials per **minute**. The Y axis | Derived and denormalized; nothing enforces the identity, so a stored value can disagree with a recomputation. |
| `completed_trials` | INTEGER | Trials that finished in this cell | **NULL ≠ 0.** A cell where everything failed (0) is materially different from one never measured (NULL), and the throughput numerator silently differs. |
| `p95_trial_latency_s` | REAL | 95th-percentile per-trial latency, seconds | A **percentile** — cannot be averaged across replicates or cells to get a valid overall p95. Documented defect: persisted as `0.0` on every real sweep today (§8). |
| `cpu_avg` | REAL | Node-level mean CPU **percent** over the cell window | Rolled up from the **`l1_system`** measurement layer, i.e. copied out of `measurements.payload`. Denormalized — the source rows may change independently. **Node-wide**, so on a shared machine it includes other tenants' load. |
| `cpu_p95` | REAL | p95 node CPU percent | Percentile — not averageable. |
| `cpu_peak` | REAL | Peak node CPU percent | A single sample's max — highly sensitive to the `l1_system` sampling interval; not comparable across sweeps with different sample rates. |
| `runqueue_max` | REAL | Max run-queue length — the primary saturation/knee signal | REAL for a count because it comes from a sampled average series. Interpretable only against `sweeps.vcpu_basis` (`runqueue_max >> vcpu_basis` = oversubscribed). |
| `ctx_sw_per_s` | REAL | Context switches per **second** | The payload key is `ctx_sw_per_s_avg` — it is a mean over the window, not an instantaneous rate. |
| `mem_avail_mb_min` | REAL | Minimum available memory, **megabytes** | MB, not KB (contrast `measurements.rss_kb_peak`). A **minimum**, so averaging across replicates is meaningless. |
| `iowait_pct_avg` | REAL | Mean iowait, percent of CPU time | NULL ≠ 0. |
| `metadata` | JSON | Keys outside the 15 promoted ones | `'{}'` not NULL; clobbered on upsert. Per-task detail (`per_task_points`) is **not** stored here — it only feeds the in-memory `ScalingAnalyzer` call. |

**Constraints.** `FOREIGN KEY (sweep_id) REFERENCES sweeps(sweep_id)` with **`NO
ACTION`** — deleting a `sweeps` row while its points exist would fail, which is why
`delete_sweep` deletes `sweep_points` first. There is deliberately **no FK on
`run_id`**. Index `idx_sweep_points_density(sweep_id, density)`.

**Introduced by.** `schema.sql`. Never altered by any migration.

**Writers.** `store_sweep_point` (17-column upsert on `(sweep_id, run_id)`
overwriting every non-key column), always **after** `_ensure_run_row` for that cell.
**Commits directly**, not via `self._commit()`.

**Readers.** `query_sweep_points(sweep_id)` → `SELECT * WHERE sweep_id = ? ORDER BY
density, replicate`; `ScalingAnalyzer.analyze_sweep`; `demo_app.py` and
`live_dashboard.py` Concurrency Sweep pages.

**Why it exists.** The pre-aggregated scaling curve. The knee analysis needs
throughput and node-saturation signals per operating point, and deriving those on the
fly would mean re-scanning every cell run's `l1_system` measurement rows on every
dashboard render. It is the **only** place the cell-level rollup lives, so losing it
loses the sweep even though the per-cell runs still exist. An internal UX assessment
notes the dashboard's sweep page reads this table and is therefore unaffected by the
planned sweep-runner rewrite **as long as the replacement keeps writing here** —
making this table's shape a de-facto stable contract.

---

### 3.10 Index inventory

| Index | On | Serves |
|---|---|---|
| `idx_runs_benchmark` | `runs(benchmark_id)` | `list_runs(benchmark_id=)`, `latest_run`, `delete_benchmark`, `list_benchmarks` join. Compensates for `benchmark_id` not being an FK (no implicit index). |
| `idx_runs_owner` | `runs(owner_id)` | `list_runs(owner_id=)` — "my experiments" vs the shared view. |
| `idx_task_logical` | `task_results(logical_task_key)` | **The** cross-run aggregation index. |
| `idx_task_workload` | `task_results(workload_type)` | `query_tasks(workload_type=)`. Useless in practice — the column is NULL on every real row. |
| `idx_meas_run_layer` | `measurements(run_id, layer)` | The commonest measurement access pattern; `cli db show`'s per-layer coverage. |
| `idx_meas_span` | `measurements(run_id, span_id)` | Per-span drilldown across layers. |
| `idx_meas_run_task` | `measurements(run_id, task_id)` | Per-task rollup matching the exporter's label set. |
| `idx_spans_task` | `spans(run_id, task_id)` | All steps of one task. |
| `idx_spans_kind` | `spans(run_id, span_kind)` | `span_kind='llm_call'` — the token/cost path. |
| `idx_verdict_analyzer` | `analyzer_verdicts(analyzer_name, verdict)` | Cross-run "how many runs were io_bound". Does **not** lead with `run_id`. |
| `idx_artifacts_run_kind` | `artifacts(run_id, kind)` | Resolving a run's EMON CSV / plot back to a path. |
| `idx_sweep_points_density` | `sweep_points(sweep_id, density)` | The scaling-curve read. |

Plus the implicit UNIQUE autoindexes on every PK / UNIQUE constraint, each of which
also serves bare `WHERE run_id = ?` by left prefix.

**Not present — know these before you write a query.**

- **No index on `runs(start_time)`**, despite `list_runs`' `ORDER BY start_time DESC,
  run_id DESC LIMIT ?` being the primary discovery query. Fine at today's row count; a
  sort-on-scan that degrades as runs accumulate.
- **No JSON1 expression indexes** over `measurements.payload`. All indexes are
  origin `c`/`pk`/`u`. Filtering an unpromoted payload key is a full scan +
  `json_extract` — keep it off hot paths or push it to DuckDB. (**PLANNED**:
  `STORAGE_UNIFICATION_PLAN.md` line 246.)
- **No VIEWs of any kind.** `sqlite_master` has no `type='view'` rows on any DB.
  There is no stable projection layer inside the database to hide 0003/0007-style
  reshapes — §7 gives you one to install on your own connection.

---

## 4. Identity and aggregation keys

| Question | Correct key | Why the obvious choice is wrong |
|---|---|---|
| Same task across N runs | `task_results.logical_task_key` (`'{benchmark_id}::{task_id}'`, indexed) | Bare `task_id` is unique only *within* a benchmark, and before 0003 it was a **global PK** so run N+1 destroyed run N. |
| Same benchmark across N runs | `runs.benchmark_id` | `run_id` is per-execution and can never group anything. Mind the slug normalisation: plugin `synthetic_cpu` → stored `synthetic-cpu`. |
| A sweep across its cells | `(sweeps.sweep_id, sweep_points.density)`, averaging over `replicate` | Grouping by `run_id` treats each replicate as its own operating point. Do **not** parse density out of the cell run_id — `%g` turns `1.0` into `d1`. |
| Measurement dedup identity | `(run_id, span_id, layer)` | One task fans out to many spans, and one span to many layers. |
| Verdict identity | `(run_id, task_id, analyzer_name)` | There is not one verdict per run. |
| Cross-hardware comparison | `(benchmark_id, logical_task_key, hardware_sku, host_id, optimization_profile, numa_policy)` | `hardware_sku` alone does not distinguish two nodes of the same SKU, and the real value is a raw CPUID string, which on some hosts is a generic placeholder shared by every part of that stepping. |

### `run_id`

A **handle, not a fingerprint**. The documented invariant is byte-identity: one
`run_id` string, identical across the store, the Grafana label and the Langfuse
`trace_user_id`, with user/host/benchmark scope **never** folded into it — owner
identity is a *column* (`owner_id`/`owner_kind`/`host_id`), because folding it into
the id would desync the Pushgateway and Langfuse sides
(`docs/STORAGE_IMPLEMENTATION_PLAN.md` invariant 1). ULID run_ids are **PLANNED**
(`STORAGE_UNIFICATION_PLAN.md` §1); shipped ids are `f'{benchmark}_{stamp}'` or
`f'run-{uuid4().hex[:8]}'`, so **`ORDER BY run_id` is not chronological**.

### `span_id` shapes — there is no single grammar

`span_id` round-trips **raw and byte-exact**, and it is the *only* key joining
hardware measurements to token/cost data. Two shapes coexist by design, and more
exist in practice:

| Shape | Producer | Derived `measurements.task_id` |
|---|---|---|
| `compile` (bare task) | trace builder parent span | `compile` |
| `{task}/turn_{i}_llm`, `{task}/turn_{i}_cmd` | trace builder per-turn | `linalg/turn_0_llm` — **the whole string**, so it never joins |
| `{task}::{phase}` | six-phase adapter | `{phase}` |
| `{run_id}::emon_run` | EMON probe | `emon_run` — a pseudo-task |
| `{run_id}::cell` | sweep driver | `cell` — a pseudo-task |
| `swe/{instance}/step_{i}_llm`, `tau/{task}/turn_{i}_{phase}` | EMON examples | the whole string |

The store's rule is `span_id.split('::')[-1] if '::' in span_id else span_id`. Do
**not** invent your own split rule: the store, the Prometheus exporter and
`demo_app` all use this exact line, and diverging makes three consumers disagree.

### The three `task_id` columns are not the same thing

| Column | Derivation | Safe to join to `task_results.task_id`? |
|---|---|---|
| `task_results.task_id` | the adapter's real task id | — (it *is* the reference) |
| `measurements.task_id` | `span_id.split('::')[-1]` or `span_id` | **No** |
| `spans.task_id` | `parent_span_id or span_id` | **No** |
| `analyzer_verdicts.task_id` | `AnalysisResult.span_id or 'unknown'` | **No** |

Joining any two of these appears to work on the live DB **only** because its
synthetic span_ids happen to be bare task names. The views in §7 expose these under
distinct names (`derived_task_id`, `span_task_id`, `verdict_scope_id`) and give you
count-based bridges plus a `verdict_scope` classifier instead of a false join.

### `sweep_id`

One string, three roles: `sweeps.sweep_id`, a **placeholder `runs.run_id`** (so the
scaling verdict has an FK parent), and `analyzer_verdicts.run_id`. Every aggregate
must exclude sweep and sweep-cell rows or a 4-density × 2-replicate sweep silently
adds 9 rows that look like ordinary runs.

---

## 5. Migrations and versioning

**Mechanism.** `PRAGMA user_version` plus ordered `storage/migrations/000N_*.sql` —
"alembic-lite", zero new dependency. Alembic arrives only if Postgres does.

**How the runner works** (`src/storage/sqlite_store.py`):

1. `_ensure_db()` unconditionally `executescript(schema.sql)` **first** (all
   `IF NOT EXISTS`, so a no-op on an existing DB), commits, then `_run_migrations()`.
2. `_run_migrations` reads `PRAGMA user_version`, globs `migrations/[0-9]*.sql`,
   `sorted()` lexicographically, parses the version as `int(name.split("_",1)[0])`;
   unparseable names are logged and skipped.
3. **Baseline adoption.** If `current == 0` and any migration has `v <= 1`, it stamps
   `user_version = 1` **without executing that file's DDL** — the rationale being
   that `schema.sql` already produced the baseline shape. `0001_baseline.sql` is a
   pure comment marker with zero DDL, which is what makes this safe. This is also
   how pre-versioning legacy DBs get adopted.
4. Each migration with `version > current` then runs:
   `PRAGMA foreign_keys = OFF` (in autocommit — the pragma is a no-op inside a
   transaction) → `executescript(sql)` → `PRAGMA user_version = N` → `commit()` →
   `PRAGMA foreign_key_check` as the integrity gate → `finally: PRAGMA foreign_keys
   = ON`. Any `sqlite3.Error` is wrapped in `RuntimeError` and re-raised —
   **fail-loud, never swallowed** (abort-don't-degrade).

So a fresh DB is: `schema.sql` shape → stamped v1 → 0002…0008 applied → **v8**.
Verified on a freshly created DB under a scratch `AGENTSYSPERF_HOME`. A store that
predates 0008 reports `user_version = 7` until a writer opens it; a read-only open
never migrates (§6), so a consumer can legitimately encounter either version.

**The table-rebuild pattern SQLite forces.** SQLite cannot `ALTER` a primary key or
add a constraint. Migrations 0003 and 0005 therefore: backfill placeholder `runs`
rows for any dangling children (so the new cascade FK loses nothing) → `CREATE TABLE
x_new` with the target shape → `INSERT … SELECT` (0003 writes `logical_task_key` as
NULL because no benchmark is knowable at migration time; 0005 dedups by keeping
`MAX(verdict_id)` per identity) → `DROP TABLE x` → `ALTER TABLE x_new RENAME TO x`
→ recreate the indexes that died with the old table. Both embed their own
`BEGIN; … COMMIT;` **inside the script text**, because `sqlite3.executescript()`
implicitly commits any pending transaction, making a Python-side `BEGIN` a no-op.

**Migration table.**

| # | File | What | Why |
|---|---|---|---|
| 0001 | `0001_baseline.sql` | Comment-only marker, zero DDL | Establishes that a fresh DB (built by `schema.sql`) is at v1 so 0002+ apply cleanly, and lets pre-versioning DBs be adopted without re-running baseline DDL. |
| 0002 | `0002_measurements.sql` | Creates `measurements` with 6 promoted columns, `payload` JSON, `seq`, `UNIQUE(run_id, span_id, layer)`, cascade FK, 3 indexes | Ends the split-brain: `store_measurements` was a **no-op**, so all L1/L3/EMON/perfspect data lived only in per-directory JSON and the DB was not the system of record for hardware data. |
| 0003 | `0003_task_results_composite_pk.sql` | Rebuilds `task_results`: composite PK `(run_id, task_id)`, adds `logical_task_key`, adds `ON DELETE CASCADE`, recreates indexes | Fixes a data-loss bug: the global `task_id` PK meant the same task under two run_ids collided and the second overwrote the first, making cross-run aggregation and variance measurement impossible. |
| 0004 | `0004_benchmarks_and_provenance.sql` | Adds `benchmarks`; nine additive `ALTER TABLE runs ADD COLUMN` (benchmark_id, optimization_profile, numa_policy, agentperf_version, result_digest, owner_id, owner_kind, host_id, status); `idx_runs_benchmark`, `idx_runs_owner` | Adds the missing top-level entity and promotes provenance out of free-form JSON so a published number is defensible and multi-user attribution works. `benchmark_id` is intentionally **not** an FK. |
| 0005 | `0005_verdict_unique.sql` | Rebuilds `analyzer_verdicts` with `UNIQUE(run_id, task_id, analyzer_name)` + cascade FK; dedups on copy; backfills placeholder runs for dangling sweep verdicts | `store_analysis_results` did a plain INSERT, so re-running analyzers multiplied verdict rows and inflated every dashboard count. |
| 0006 | `0006_artifacts.sql` | Adds `artifacts` (pointer registry) + `UNIQUE(run_id, kind, name)` + cascade FK + `idx_artifacts_run_kind` | Bulk per-run files stay on disk but must be addressable **by run_id** instead of rediscovered by globbing scattered `/tmp` dirs. |
| 0007 | `0007_rename_agentsysperf_version.sql` | `ALTER TABLE runs RENAME COLUMN agentperf_version TO agentsysperf_version` | 0004 shipped the column as `agentperf_version` while the store writes `agentsysperf_version`, so **every** DB failed `persist_run` with *"no column named agentsysperf_version"* and rolled back the whole run write. Regression-pinned by `test_p3_schema.py::test_agentsysperf_version_persists`. |
| 0008 | `0008_drop_hardware_sku_default.sql` | Rebuilds `runs` to remove the `hardware_sku` DEFAULT; copies every column and row verbatim; recreates `idx_runs_benchmark`, `idx_runs_owner` | `schema.sql` defaulted the SKU to `'Intel Xeon Platinum 8592+'` — the authoring host's silicon — so any run that omitted a SKU was silently labelled Emerald Rapids. For a cross-platform benchmark, a fabricated hardware label is the worst possible default: wrong in a way that looks like data. An unknown SKU is now NULL ("not recorded"). Existing rows are preserved, not guessed at. |

**Two runner caveats a consumer must know.**

1. **The migration runner is not atomic across the FK gate, and its error message
   lies.** It executes the script, sets `user_version = N`, **commits**, and only
   then runs `PRAGMA foreign_key_check`. If the check finds violations, the handler
   calls `rollback()` (a no-op — already committed) and raises
   `RuntimeError('Migration X failed (DB left at version {current})')`. Reproduced
   with a synthetic FK-violating migration: the exception said "left at version 1"
   while `user_version` was actually 2, the new table existed, and the violating row
   was still there. **A failed migration can leave the DB versioned as successful
   with bad data, and the next open skips it.**
2. **Migration numbering has no collision detection.** `0007` has two *file*
   claimants — the committed `0007_rename_agentsysperf_version.sql` on main versus
   `0007_agent_side_metric.sql` in PR #5 — plus a third, still-unnumbered `0007`
   proposal in an internal scaling-visualization plan. (PR #5's own rename is
   numbered **0008**, not a second `0007`.) With two `0007` files present,
   `sorted()` applies whichever sorts first (`…agent_side…` < `…rename…`) and
   **silently skips** the other, because the skipped file's version is no longer
   `> current`. Both outcomes were reproduced by dropping PR #5's two files into
   `migrations/`:
   - **An existing v7 DB will not open at all**:
     `RuntimeError: Migration 0008_rename_agentperf_version_column.sql failed
     (DB left at version 7): no such column: "agentperf_version"` — because the
     committed 0007 already performed that rename. Worse than a silent skip: the
     store is unreachable until the migration set is renumbered.
   - **A fresh DB lands at `user_version = 8`** with the correct shape
     (`runs.agentsysperf_version` plus `task_results.agent_side_s`/`censored`),
     because the skipped `0007` rename is redundant with PR #5's `0008`. So the
     same migration set produces a working fresh DB and a bricked existing one.

   Consequence for you: **`PRAGMA user_version` is necessary but not sufficient** —
   also probe for the specific columns you read (§7.2).

---

## 6. Connection semantics

**One cached connection per store object.** `_get_connection()` lazily creates a
single `sqlite3.Connection` on `self._conn` and reuses it for the object's life,
with `check_same_thread=False` and `row_factory = sqlite3.Row` on both branches.
The class docstring claims thread-safety for reads, but **there is no lock around
the shared connection** — concurrent writes from multiple threads are the caller's
problem.

| | Read-write open | Read-only open |
|---|---|---|
| connect | `sqlite3.connect(str(db_path))` | `sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)` |
| `journal_mode` | `WAL` | not set — and don't. `PRAGMA journal_mode=WAL` on a read-only handle **succeeds** as a no-op (returns `('wal',)`; the mode already persists in the file header), so a success here is **not** evidence the handle is writable. Only a mode *change* fails: `PRAGMA journal_mode=DELETE` → `OperationalError: disk I/O error`. |
| `foreign_keys` | `ON` | `ON` |
| DDL / migrations | yes | **never** |

**`PRAGMA foreign_keys` is per-connection and OFF by default in SQLite.** Re-setting
it on every new connection is load-bearing: without it every `ON DELETE CASCADE` in
this schema is decorative and `delete_run` would silently orphan children. Pinned by
`test_measurements_roundtrip.py::test_p0_pragmas_and_version`, which asserts
`foreign_keys == 1`, `journal_mode == 'wal'` and `user_version >= 1`. **An external
consumer opening the file with its own `sqlite3` connection gets no referential
integrity unless it sets the pragma itself.**

**WAL.** The mode persists in the DB file header once set, so `-wal` and `-shm`
sidecars sit next to `results.db`. WAL gives one writer plus N concurrent readers,
which is the entire no-daemon design. Two consequences:

- **Never `cp results.db` to snapshot it** — you will silently lose the WAL tail,
  i.e. the most recent runs. Use `VACUUM INTO '/path/snapshot.db'` or the backup API
  from a read-only connection.
- A long-lived read connection pins a snapshot and prevents the writer from
  checkpointing past your read mark. Open per request (or per cache refresh), read,
  close — the pattern `dashboard_data.py` already uses, caching plain dicts and
  lists, **never a connection**.

**Read-only stores never migrate.** `__init__` calls `_ensure_db()` only when
`read_only=False`, so a viewer opening a stale file sees whatever schema is on disk
with **no upgrade and no warning** (pinned by
`test_open_seam.py::test_readonly_open_does_not_create_db`). The docstring frames
this as a safety feature, which it is — but there is no version check telling the
viewer it is looking at a v4 shape. Feature-detect (§7.2).

Conversely, **auto-migrate-on-open is in-place and one-way**: opening a DB as a
*writer* upgrades it. Never point a writer at someone else's DB
(`STORAGE_IMPLEMENTATION_PLAN.md`, DECISIONS 2026-06-11).

### Transaction discipline

Two tiers, and they are **inconsistent**:

- `_commit()` commits **unless** `self._in_transaction`, which `persist_run` sets.
  Used by `store_artifact`, `store_run_metadata`, `store_task_result`,
  `store_measurements`, `store_spans`, `store_analysis_results`.
- `_on_write_error()`: a standalone call logs a warning and rolls back (degrade);
  **inside `persist_run` the exception is re-raised** so the whole transaction rolls
  back — abort-don't-degrade, no half-written run.
- `persist_run` sets `_in_transaction = True`, calls the `store_*` methods **in
  dependency order** (metadata first, so both the FK parent and
  `store_task_result`'s `benchmark_id` lookup resolve), then one `conn.commit()`;
  any exception → `rollback()` + re-raise; `finally` resets the flag. It covers
  `runs`, `task_results`, `measurements`, `spans`, `analyzer_verdicts` — **not**
  `benchmarks`, `sweeps`, `sweep_points` or `artifacts` (though `store_artifact`
  itself is `persist_run`-safe).
- **The split:** `store_benchmark`, `store_sweep_metadata`, `store_sweep_point`,
  `delete_run`, `delete_benchmark` and `delete_sweep` call
  `conn.commit()`/`rollback()` **directly**, bypassing `_commit`. None is currently
  called from inside `persist_run`, so this is latent, not live — but adding such a
  call would break `persist_run`'s atomicity with no test catching it.

### DSN forms and where the database lives

Resolution order: explicit `dsn=` argument > `$AGENTSYSPERF_STORE_DSN` >
`$AGENTSYSPERF_STORE_URL` > `$AGENTSYSPERF_HOME/results.db` >
`~/.agentsysperf/results.db`. Explicitly **not** `/tmp` — results must survive a
reboot.

| DSN | Resolves to | Note |
|---|---|---|
| `/abs/path.db` | that file | bare path, no `://` |
| `sqlite:///rel/db` | `rel/db` | urlparse `.path` = `/rel/db`, one leading slash stripped → **RELATIVE to cwd**. Footgun. |
| `sqlite:////abs/db` | `/abs/db` | `.path` = `//abs/db`, `[1:]` → absolute. **Four slashes required for an absolute path.** |
| anything else | `NotImplementedError` naming the scheme | the deliberate seam for a future server backend |

Artifacts live outside the DB under `$AGENTSYSPERF_HOME/artifacts/`; per-run scratch
defaults to `/tmp/agentsysperf_scratch/{run_id}`.

**The two-filename trap.** `open()` writes `results.db`. The legacy
`SQLiteResultStore(output_dir=X)` constructor writes
`X/agentsysperf_results.db` — a **different file in the same directory**. Point the
constructor at a directory the CLI wrote and you get an **empty second DB** where
every query returns zero rows with no error, which reads as "the run recorded
nothing". The constructor detects exactly that case and logs a WARNING rather than
switching, because the dashboards intentionally open the legacy name to find old
sweeps (pinned by
`test_open_seam.py::test_output_dir_warns_when_it_shadows_a_canonical_store`).
**Pin your path explicitly.** Reports of a legacy `~/.agentperf/results.db` are
stale: no such file exists and no code path looks for it — only the *column* was
renamed, by migration 0007.

### DuckDB read side

`attach()` runs `INSTALL sqlite; LOAD sqlite; ATTACH '{path}' AS perf (TYPE sqlite,
READ_ONLY);` — zero copy, SQLite stays the single system of record, tables reachable
as `perf.runs`, `perf.measurements`, and so on. `export_parquet(out,
table='measurements')` runs `COPY (SELECT * FROM perf.{table}) TO '…' (FORMAT
parquet)` — note both `table` and the path are f-string-interpolated with no
allowlist or quoting.

---

## 7. Consuming results from another project

### 7.1 Stability tiers

**STABLE — safe to depend on.** Names, declared types, units and key semantics will
not change without a migration plus a version bump. Depend on them **by name**,
never by ordinal position (0003 physically reordered `task_results`; 0007 renamed a
`runs` column). NULL-vs-zero semantics documented above are part of the promise.

- `runs`: `run_id`, `start_time`, `end_time`, `status`, `benchmark_id`,
  `hardware_sku`, `host_id`, `owner_id`, `owner_kind`, `model`,
  `optimization_profile`, `numa_policy`, `agentsysperf_version`, `metadata`
- `task_results`: the `(run_id, task_id)` PK, `logical_task_key`, `passed`,
  `duration_s`
- `measurements`: the `(run_id, span_id, layer)` UNIQUE identity, `layer`, `span_id`,
  `payload`, the six promoted columns (stable names and **units**, NULL-by-layer),
  `seq`
- `analyzer_verdicts`: the `(run_id, task_id, analyzer_name)` UNIQUE identity,
  `verdict`, `confidence`, `evidence`, `recommendations`
- `sweeps` / `sweep_points`: the columns listed in §3.8–3.9
- `SQLiteResultStore.open(dsn=…, read_only=True)` and `from_dsn()`
- The explicitly-projected read methods: `query_tasks`, `query_measurements`,
  `query_verdicts`, `list_artifacts`, `list_benchmarks`
- The eight `v_*` views in §7.3 — the intended external projection layer

**INTERNAL — no promise; do not persist or display.**
`measurement_id` / `verdict_id` / `artifact_id`; all three derived `task_id`
columns; the five unrelated "version" concepts; `runs.total_tasks` /
`passed_tasks` (denormalized, dead defaults); the `hardware_sku` DDL default;
`artifacts.created_at` / `metadata` (unwritable); the `artifacts.kind` vocabulary;
the `SELECT *` methods `list_runs` / `get_run` / `query_spans` / `query_sweeps` /
`query_sweep_points` (the API is stable, the **row shape** is an unversioned
contract); the second DB filename; the sweep_id/run_id namespace overload;
migration-backfilled placeholder runs; Prometheus/Grafana metric names and Langfuse
trace shapes.

**EXPERIMENTAL — schema-stable, data sparse. Degrade gracefully.**
`spans` (**sparse, not empty**: `store_spans` is only wired on the TB2/agent-loop
path, so a TB run carries a full step trace — measured 52 spans, 440k prompt tokens,
$0.62 on one run — while every token/cost/step panel is empty for CLI and synthetic
runs. Check `v_run_header.span_rows` per run rather than assuming either state);
`spans.tokens_in` /
`tokens_out` / `cost_usd` / `duration_us` / `status` (defaults make "unmeasured"
indistinguishable from a real zero, and an unset outcome reads as success);
`sweeps` / `sweep_points` (0 rows — sweeps still write to `/tmp`);
`sweep_points.concurrency` (requested ceiling, not achieved),
`p95_trial_latency_s` (persisted 0.0), `sweeps.numa_policy` (stored, never applied);
`benchmarks` (0 rows — always UNION the run-derived slugs); `artifacts` (0 rows);
`task_results.workload_type` / `num_turns` / `num_commands` (NULL everywhere);
layers `emon` / `perfspect` (need external Intel binaries) and `l1_system`
(undocumented in the DDL comment, a third of real rows, populates **zero** promoted
columns); `runs.result_digest` (NULL everywhere); `duckdb_analytics` (optional
extra, unmeasured crossover). **Render "no data", not zero, and never compute a rate
or total from these without checking a sample count.**

**PLANNED — does not exist. Feature-detect, never assume.**
ULID `run_id`s and `runs.run_label` (`STORAGE_UNIFICATION_PLAN.md` §1) — so
`ORDER BY run_id` is not chronological; soft-delete tombstones
`runs.deleted_at` / `reason` / `status='deleted'` (§3) — `delete_run` is a hard
delete; `measurements.node_id` and `measurements.kind` as **columns** (lines 59–60)
— they exist only as payload keys; JSON1 expression indexes (line 246); any SQL VIEW
in the database; the `sweeps`/`sweep_points` widening in
an internal scaling-visualization plan; a Postgres backend, the `tar.zst`
export/import bundle, an `ArtifactStore` local|s3 shim, `agentsysperf serve` /
`migrate` subcommands, a retention/compaction policy; `runs.code_git_sha`,
`dataset_version`, `replay_fixture_hash`, `env_capture`.

### 7.2 The correct read-only access pattern

**The pragma ORDER is load-bearing.** `PRAGMA query_only = 1` and
`CREATE TEMP VIEW` are mutually exclusive: `query_only` gates the *temp* schema too,
not just the main DB, so setting it before installing the §7.3 views makes every one
of the eight fail with `sqlite3.OperationalError: attempt to write a readonly
database`. Verified trial matrix on SQLite 3.45.1 against
`file:…?mode=ro`: `query_only=1` then `CREATE TEMP VIEW` → FAIL (with or without
`foreign_keys=ON`, and `temp_store=MEMORY` is **not** a workaround); no `query_only`
→ OK. Install the views **first**, set `query_only` **last**.

```python
import os, sqlite3, pathlib

# $AGENTSYSPERF_HOME/results.db, defaulting to ~/.agentsysperf/results.db —
# resolve it, never hardcode a home directory.
DB = pathlib.Path(
    os.environ.get("AGENTSYSPERF_HOME", pathlib.Path.home() / ".agentsysperf")
) / "results.db"

conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)  # URI form is MANDATORY:
                                                        # a plain path would create
                                                        # and/or upgrade the file
conn.row_factory = sqlite3.Row
conn.execute("PRAGMA foreign_keys = ON")   # per-connection; OFF by default, so
                                           # without this every cascade in the
                                           # schema is decorative on your handle
# Do NOT set journal_mode on a read-only handle. '=WAL' silently SUCCEEDS as a
# no-op (the mode lives in the file header), so it proves nothing; only a mode
# change fails, with 'disk I/O error'.

# Feature-gate, because a read-only open NEVER migrates.
(user_version,) = conn.execute("PRAGMA user_version").fetchone()
assert user_version >= 7, f"stale schema: v{user_version} (agentsysperf_version was named agentperf_version before migration 0007)"
# v8 dropped the fabricated runs.hardware_sku default. On a v7 store a NULL SKU may
# instead be the literal 'Intel Xeon Platinum 8592+' — do not trust it as measured.
sku_is_trustworthy = user_version >= 8

def has_table(c, name):
    return c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                     (name,)).fetchone() is not None

def has_column(c, table, col):
    return any(r["name"] == col for r in c.execute(f"PRAGMA table_info({table})"))

# artifacts arrived in 0006; user_version alone is not sufficient because
# migration numbering has no collision detection (see §5).
assert has_table(conn, "artifacts")
assert has_column(conn, "runs", "agentsysperf_version")

# Install the contract views on THIS connection only (TEMP = per-connection temp
# schema, needs no write access to the main DB). agentsysperf_views.sql is a file
# YOU create by copying the eight CREATE TEMP VIEW blocks out of §7.3 into your
# own repo — this project does not ship it.
conn.executescript(pathlib.Path("agentsysperf_views.sql").read_text())

# ONLY NOW, after the views exist. Setting this any earlier breaks the line above.
conn.execute("PRAGMA query_only = 1")
```

Verified end to end: `mode=ro` → `foreign_keys=ON` → all eight views created →
`query_only=1` yields `v_run_header` rows and still rejects a subsequent
`CREATE TEMP VIEW` with *attempt to write a readonly database*.

Open per request or per cache refresh, read, close. Never hold the connection in a
cache — cache the plain dicts and lists you read out of it.

### 7.3 The contract views

The database contains **zero** views. Do not add permanent ones to the shared file:
that is a write, it needs the writable connection, and it puts your contract inside
someone else's system of record. Instead run these eight `CREATE TEMP VIEW`
statements on each read-only connection after opening it. Ship them as one `.sql`
file in **your** repo and version it independently, so a reshape like 0003 or 0007
is absorbed in one place instead of across every page.

All eight were executed against a `VACUUM INTO` snapshot of
`/home/<user>/.agentsysperf/results.db` opened `mode=ro` — **with `query_only`
deferred until after the views exist**, per §7.2, since `query_only=1` blocks
`CREATE TEMP VIEW` outright. For the one whose tables are empty there
(`v_sweep_density`) they were additionally run against a purpose-built store
containing a real sweep. `v_logical_task_summary` uses window functions, so
**SQLite ≥ 3.25** is required (measured 3.45.1 in the project venv).

The **column count** per view is stable and is stated below. **Row counts are not**
— the store is live (see the header note), so each view's row count is stated only
as of 2026-07-30 and will differ for you.

#### 7.3.1 `v_run_header` — individual run detail

Identity, provenance, a **computed** pass rate, and a coverage manifest telling the
page which panels have data. Also the safe "recent runs" source, because it excludes
the sweep, sweep-cell and migration-backfilled placeholder rows that would otherwise
flood a run list.

*Grain:* one row per real benchmark run. **31 columns.** *Verified:* executes on the
live DB (10 rows as of 2026-07-30); on a store with a sweep (12 `runs` rows = 3 real
+ 1 sweep placeholder + 8 cells) it correctly returns 3.

```sql
CREATE TEMP VIEW v_run_header AS
SELECT
  r.run_id,
  r.benchmark_id,
  b.display_name                                  AS benchmark_display_name,
  b.version                                       AS benchmark_version,
  r.start_time,
  r.end_time,
  CASE WHEN r.end_time IS NOT NULL AND r.start_time IS NOT NULL
            AND r.end_time >= r.start_time
       THEN r.end_time - r.start_time END          AS wall_s,
  r.status,
  r.hardware_sku,
  r.host_id,
  r.owner_id,
  r.owner_kind,
  r.model,
  r.optimization_profile,
  r.numa_policy,
  r.agentsysperf_version,
  r.result_digest,
  r.total_tasks                                    AS total_tasks_reported,
  r.passed_tasks                                   AS passed_tasks_reported,
  (SELECT COUNT(*) FROM task_results t WHERE t.run_id = r.run_id)                  AS tasks_observed,
  (SELECT COUNT(*) FROM task_results t WHERE t.run_id = r.run_id AND t.passed = 1) AS tasks_passed_observed,
  CASE WHEN (SELECT COUNT(*) FROM task_results t WHERE t.run_id = r.run_id) > 0
       THEN 1.0 * (SELECT COUNT(*) FROM task_results t WHERE t.run_id = r.run_id AND t.passed = 1)
                / (SELECT COUNT(*) FROM task_results t WHERE t.run_id = r.run_id)
  END                                              AS pass_rate_observed,
  CASE WHEN r.total_tasks IS NULL THEN NULL
       WHEN r.total_tasks = (SELECT COUNT(*) FROM task_results t WHERE t.run_id = r.run_id) THEN 1
       ELSE 0 END                                  AS task_count_agrees,
  (SELECT COUNT(*) FROM measurements m WHERE m.run_id = r.run_id)                  AS measurement_rows,
  (SELECT group_concat(l.layer) FROM (SELECT DISTINCT layer FROM measurements m
        WHERE m.run_id = r.run_id ORDER BY layer) l)                               AS measurement_layers,
  (SELECT COUNT(*) FROM spans s WHERE s.run_id = r.run_id)                         AS span_rows,
  (SELECT COUNT(*) FROM analyzer_verdicts v WHERE v.run_id = r.run_id)             AS verdict_rows,
  (SELECT COUNT(*) FROM artifacts a WHERE a.run_id = r.run_id)                     AS artifact_count,
  (SELECT json_group_array(json_object('kind', a.kind, 'name', a.name, 'path', a.path))
     FROM artifacts a WHERE a.run_id = r.run_id)                                   AS artifacts_json,
  CASE WHEN json_valid(r.metadata) THEN r.metadata END                             AS metadata_json,
  CASE WHEN json_valid(r.metadata) THEN json_extract(r.metadata, '$.run_kind') END AS run_kind
FROM runs r
LEFT JOIN benchmarks b ON b.benchmark_id = r.benchmark_id
WHERE NOT EXISTS (SELECT 1 FROM sweeps sw       WHERE sw.sweep_id = r.run_id)
  AND NOT EXISTS (SELECT 1 FROM sweep_points sp WHERE sp.run_id  = r.run_id)
  AND COALESCE(CASE WHEN json_valid(r.metadata)
                    THEN json_extract(r.metadata, '$.run_kind') END, '')
      NOT IN ('sweep', 'sweep_cell')
  AND COALESCE(CASE WHEN json_valid(r.metadata)
                    THEN json_extract(r.metadata, '$.backfilled_orphan') END, 0)
      NOT IN (1, 'true');
```

`total_tasks_reported` and `tasks_observed` are both exposed, plus
`task_count_agrees`, so a surface can **show** the discrepancy rather than average it
away. `benchmark_id` may be NULL (RunContext-only runs and several example scripts
never set it) — such a run is displayable but not aggregatable.

#### 7.3.2 `v_run_task` — per-task results table

*Grain:* one row per (run_id, task_id) — one attempt. **18 columns.** *Verified:*
executes on the live DB (30 rows as of 2026-07-30), `agg_key_source` all `stored`.

```sql
CREATE TEMP VIEW v_run_task AS
SELECT
  t.run_id,
  r.benchmark_id,
  t.task_id,
  t.logical_task_key,
  COALESCE(t.logical_task_key,
           CASE WHEN r.benchmark_id IS NOT NULL THEN r.benchmark_id || '::' || t.task_id END) AS agg_key,
  CASE WHEN t.logical_task_key IS NOT NULL THEN 'stored'
       WHEN r.benchmark_id  IS NOT NULL THEN 'derived'
       ELSE 'unkeyable' END                        AS agg_key_source,
  t.workload_type,
  t.passed,
  t.duration_s,
  t.num_turns,
  t.num_commands,
  r.start_time                                     AS run_start_time,
  r.hardware_sku,
  r.host_id,
  r.optimization_profile,
  r.numa_policy,
  (SELECT COUNT(*) FROM measurements m
     WHERE m.run_id = t.run_id AND m.task_id = t.task_id)                          AS measurement_rows_matched,
  (SELECT COUNT(*) FROM spans s
     WHERE s.run_id = t.run_id AND s.task_id = t.task_id)                          AS span_rows_matched
FROM task_results t
JOIN runs r ON r.run_id = t.run_id
WHERE NOT EXISTS (SELECT 1 FROM sweeps sw       WHERE sw.sweep_id = r.run_id)
  AND NOT EXISTS (SELECT 1 FROM sweep_points sp WHERE sp.run_id  = r.run_id)
  AND COALESCE(CASE WHEN json_valid(r.metadata)
                    THEN json_extract(r.metadata, '$.run_kind') END, '')
      NOT IN ('sweep', 'sweep_cell')
  AND COALESCE(CASE WHEN json_valid(r.metadata)
                    THEN json_extract(r.metadata, '$.backfilled_orphan') END, 0)
      NOT IN (1, 'true');
```

The `backfilled_orphan` clause matches `v_run_header`'s and is **required**: without
it, a DB carrying migration 0003/0005 placeholder runs returns task rows belonging to
runs that `v_run_header` excludes (reproduced on a store emulating 0003's backfill:
`v_run_header` → `['good']` while the unfiltered `v_run_task` → `[('good','t1'),
('orphan','legacy_task')]`).

`agg_key_source` is the honesty column: `stored` = the store wrote
`logical_task_key`; `derived` = it was NULL (a pre-0003 migrated row) and the view
reconstructed it; `unkeyable` = the run has no `benchmark_id`, so the attempt cannot
participate in cross-run aggregation at all. `measurement_rows_matched` /
`span_rows_matched` are **count bridges, not joins** — they tell you whether the
derived-`task_id` rules happen to line up for this row instead of pretending they
always do.

#### 7.3.3 `v_run_span_rollup` — step-trace rollup

*Grain:* one row per (run_id, `spans.task_id`, span_kind). **17 columns.**
*Verified:* executes on the live DB and is **not** empty — as of 2026-07-30 it
returns 3 rows, all from the one terminal-bench run that goes through the agent
loop: `agent_step` (1 span), `llm_call` (30 spans, `llm_tokens_in=440393`,
`llm_cost_usd=0.61696`, `model_ids='bedrock/us.anthropic.claude-haiku-4-5-…'`) and
`tool_call` (21 spans, token/cost columns NULL rather than 0). Every CLI/synthetic
run still contributes zero rows, so a page must still degrade gracefully per run.

```sql
CREATE TEMP VIEW v_run_span_rollup AS
SELECT
  s.run_id,
  s.task_id                                        AS span_task_id,
  s.span_kind,
  COUNT(*)                                         AS span_count,
  SUM(CASE WHEN COALESCE(s.status,'ok') = 'ok' AND s.error IS NULL THEN 1 ELSE 0 END) AS ok_count,
  SUM(CASE WHEN COALESCE(s.status,'ok') <> 'ok' OR  s.error IS NOT NULL THEN 1 ELSE 0 END) AS error_count,
  SUM(s.duration_us)                               AS duration_us_sum,
  SUM(CASE WHEN COALESCE(s.duration_us,0) = 0 THEN 1 ELSE 0 END) AS duration_us_zero_count,
  MIN(NULLIF(s.start_ts_us, 0))                    AS first_start_ts_us,
  MAX(NULLIF(s.end_ts_us, 0))                      AS last_end_ts_us,
  SUM(CASE WHEN s.span_kind = 'llm_call' THEN s.tokens_in  END) AS llm_tokens_in,
  SUM(CASE WHEN s.span_kind = 'llm_call' THEN s.tokens_out END) AS llm_tokens_out,
  SUM(CASE WHEN s.span_kind = 'llm_call' THEN s.cost_usd   END) AS llm_cost_usd,
  group_concat(DISTINCT s.model_id)                AS model_ids,
  group_concat(DISTINCT s.tool_name)               AS tool_names,
  MIN(s.schema_version)                            AS span_schema_version_min,
  MAX(s.schema_version)                            AS span_schema_version_max
FROM spans s
GROUP BY s.run_id, s.task_id, s.span_kind;
```

Token and cost sums are scoped to `span_kind='llm_call'` so `tool_call` rows — which
legitimately carry 0 — cannot dilute them. `duration_us_zero_count` exists because
`spans.duration_us` defaults 0 at both layers, so a zero is ambiguous;
`first_start_ts_us`/`last_end_ts_us` `NULLIF(0)` out the legacy rows predating the
timestamp fix.

#### 7.3.4 `v_run_measurement` — metrics by layer

*Grain:* one row per (run_id, span_id, layer). **33 columns.** *Verified:* executes
on the live DB (233 rows as of 2026-07-30). Measured non-NULL counts, which are the
layer-sparsity contract: `cpu_pct_mean` (and `cpu_time_s`, `rss_kb_peak`,
`duration_us`) non-NULL on 81/81 `l1` rows and 0 on every other layer; `ipc`
non-NULL on every `l3` (64/64) **and** `perfspect` (13/13) row and nowhere else;
`cache_miss_pct` non-NULL on 64/64 `l3` rows but **0/13 perfspect** — it is `l3`-only
in practice despite sharing the "hardware counter" grouping with `ipc`.

> **This view is UNSCOPED.** It is a bare `FROM measurements` with no `WHERE`
> clause, so it also returns sweep and sweep-cell measurement rows that
> `v_run_header` excludes (reproduced: on a store with a sweep, `v_run_measurement`
> returns `run_id`s `['sw','sw::d1_r0']` with `derived_task_id` `emon_run`/`cell`).
> **Join it to `v_run_header` (or apply the run-scoping clause yourself) before
> aggregating** — filtering by a single `run_id` is fine, aggregating across runs is
> not.

```sql
CREATE TEMP VIEW v_run_measurement AS
SELECT
  m.run_id,
  m.span_id,
  m.layer,
  m.task_id                                        AS derived_task_id,
  m.seq,
  m.schema_version                                 AS measurement_schema_version,
  CASE WHEN json_valid(m.payload) THEN json_extract(m.payload, '$.kind')    END AS payload_kind,
  CASE WHEN json_valid(m.payload) THEN json_extract(m.payload, '$.node_id') END AS payload_node_id,
  m.duration_us,
  COALESCE(m.duration_us / 1000000.0,
           CASE WHEN json_valid(m.payload) THEN json_extract(m.payload, '$.duration_s') END) AS duration_s_effective,
  m.cpu_time_s,
  m.cpu_pct_mean                                   AS proc_cpu_pct_mean,
  CASE WHEN json_valid(m.payload) THEN json_extract(m.payload, '$.cpu_pct_peak')    END AS proc_cpu_pct_peak,
  m.rss_kb_peak,
  CASE WHEN json_valid(m.payload) THEN json_extract(m.payload, '$.num_threads_peak') END AS proc_threads_peak,
  m.ipc,
  m.cache_miss_pct,
  CASE WHEN json_valid(m.payload) THEN json_extract(m.payload, '$.llc_miss_per_s')  END AS llc_miss_per_s,
  CASE WHEN json_valid(m.payload) THEN json_extract(m.payload, '$.branch_miss_pct') END AS branch_miss_pct,
  CASE WHEN json_valid(m.payload) THEN json_extract(m.payload, '$.cpu_avg')  END AS node_cpu_pct_avg,
  CASE WHEN json_valid(m.payload) THEN json_extract(m.payload, '$.cpu_p50')  END AS node_cpu_pct_p50,
  CASE WHEN json_valid(m.payload) THEN json_extract(m.payload, '$.cpu_p95')  END AS node_cpu_pct_p95,
  CASE WHEN json_valid(m.payload) THEN json_extract(m.payload, '$.cpu_peak') END AS node_cpu_pct_peak,
  CASE WHEN json_valid(m.payload) THEN json_extract(m.payload, '$.runqueue_avg')     END AS node_runqueue_avg,
  CASE WHEN json_valid(m.payload) THEN json_extract(m.payload, '$.runqueue_max')     END AS node_runqueue_max,
  CASE WHEN json_valid(m.payload) THEN json_extract(m.payload, '$.ctx_sw_per_s_avg') END AS node_ctx_sw_per_s_avg,
  CASE WHEN json_valid(m.payload) THEN json_extract(m.payload, '$.iowait_pct_avg')   END AS node_iowait_pct_avg,
  CASE WHEN json_valid(m.payload) THEN json_extract(m.payload, '$.mem_avail_mb_min') END AS node_mem_avail_mb_min,
  CASE WHEN json_valid(m.payload) THEN json_extract(m.payload, '$.mem_used_mb_max')  END AS node_mem_used_mb_max,
  CASE WHEN json_valid(m.payload) THEN json_extract(m.payload, '$.logical_cpus')     END AS node_logical_cpus,
  CASE WHEN json_valid(m.payload) THEN json_extract(m.payload, '$.sample_count')     END AS sample_count,
  json_valid(m.payload)                            AS payload_is_valid_json,
  m.payload                                        AS payload_json
FROM measurements m;
```

Column **prefixes encode scope**: `proc_*` is per-process (`l1` only), `node_*` is
whole-node (`l1_system` only, all lifted from payload JSON), and
`ipc`/`cache_miss_pct`/`llc_miss_per_s`/`branch_miss_pct` are hardware counters —
`ipc` from `l3` **and** `perfspect`, the other three from `l3` only. This is the whole
point of the view: without it, "average CPU" silently means a different thing per
layer, and an `ipc` mean silently blends two different instruments.
`duration_s_effective` reconciles the `l1` microseconds column with the
`l1_system`/`l3` seconds payload key. `payload_json` is retained verbatim — it is the
source of truth and the only access path to the ~53 perfspect keys (frequency, power,
per-instruction MPIs, NUMA/LLC bandwidth; **no** TMA keys reach the store, §3.3).
`emon` rows carry a `csv_path` and nothing numeric, so exclude them from any
layer-agnostic aggregate. **Always filter by layer before averaging; order within a
run by `seq`.**

#### 7.3.5 `v_run_verdict` — analyzer verdicts with scope classification

*Grain:* one row per (run_id, task_id, analyzer_name). **11 columns.** *Verified:*
executes on the live DB (235 rows as of 2026-07-30, scope distribution 99 `task` +
127 `span` + 9 `run`); on a store with a sweep, the scaling verdict filed under
`run_id = sweep_id` is correctly classified `sweep` rather than appearing as a task.
Like `v_run_measurement` this view is **unscoped** — it applies no sweep/placeholder
exclusion (it references `sweeps` only to *classify* scope), so join it to
`v_run_header` before aggregating across runs.

```sql
CREATE TEMP VIEW v_run_verdict AS
SELECT
  v.run_id,
  v.verdict_id,
  v.analyzer_name,
  v.verdict,
  v.confidence,
  v.task_id                                        AS verdict_scope_id,
  CASE WHEN v.task_id = 'unknown' THEN 'run'
       WHEN EXISTS (SELECT 1 FROM sweeps s      WHERE s.sweep_id = v.run_id AND s.sweep_id = v.task_id) THEN 'sweep'
       WHEN EXISTS (SELECT 1 FROM task_results t WHERE t.run_id = v.run_id AND t.task_id = v.task_id)   THEN 'task'
       ELSE 'span' END                             AS verdict_scope,
  CASE WHEN json_valid(v.evidence)        THEN v.evidence        END AS evidence_json,
  CASE WHEN json_valid(v.recommendations) THEN v.recommendations END AS recommendations_json,
  CASE WHEN json_valid(v.recommendations) THEN json_array_length(v.recommendations) ELSE 0 END AS recommendation_count,
  json_valid(v.evidence)                           AS evidence_is_valid_json
FROM analyzer_verdicts v;
```

`verdict_scope` is what stops a page rendering the `'unknown'` sentinel as if it were
a task name. Both JSON columns are guarded with `json_valid` so a malformed blob
yields NULL instead of raising — note the store's own `query_verdicts` decodes
`evidence` **unguarded** and will propagate `JSONDecodeError` past its
`except sqlite3.Error` handler.

#### 7.3.6 `v_benchmark_summary` — the leaderboard row

*Grain:* one row per benchmark slug, from `benchmarks` UNION `DISTINCT
runs.benchmark_id`. **28 columns.** *Verified:* executes on the live DB and returns
**2 rows** as of 2026-07-30 — `synthetic-cpu` (`run_count=8`, `pass_rate=1.0`,
`distinct_logical_tasks=9`, `ipc_avg_l3=2.373` over 22 rows,
`ipc_avg_perfspect=1.024` over 12) and `terminal-bench` (`run_count=2`,
`pass_rate=0.0`, `ipc_avg_l3=0.449` over 42 rows, `ipc_avg_perfspect=0.535` over 1,
`llm_call_count=30`, `llm_tokens_in_total=440393`, `llm_cost_usd_total=0.617`).
Both report `is_registered=0`, proving the UNION arm is load-bearing since
`benchmarks` is empty. The `llm_*` totals are NULL only for slugs whose runs have no
`spans` rows — `synthetic-cpu` here, not `terminal-bench`.

```sql
CREATE TEMP VIEW v_benchmark_summary AS
WITH bench AS (
  SELECT benchmark_id FROM benchmarks WHERE benchmark_id IS NOT NULL
  UNION
  SELECT DISTINCT benchmark_id FROM runs WHERE benchmark_id IS NOT NULL
),
scoped AS (
  SELECT r.run_id, r.benchmark_id, r.start_time, r.hardware_sku, r.host_id
  FROM runs r
  WHERE r.benchmark_id IS NOT NULL
    AND NOT EXISTS (SELECT 1 FROM sweeps sw       WHERE sw.sweep_id = r.run_id)
    AND NOT EXISTS (SELECT 1 FROM sweep_points sp WHERE sp.run_id  = r.run_id)
    AND COALESCE(CASE WHEN json_valid(r.metadata)
                      THEN json_extract(r.metadata, '$.run_kind') END, '')
        NOT IN ('sweep', 'sweep_cell')
)
SELECT
  bench.benchmark_id,
  b.display_name,
  b.version,
  CASE WHEN b.benchmark_id IS NULL THEN 0 ELSE 1 END AS is_registered,
  (SELECT COUNT(*)        FROM scoped s WHERE s.benchmark_id = bench.benchmark_id) AS run_count,
  (SELECT MIN(s.start_time) FROM scoped s WHERE s.benchmark_id = bench.benchmark_id) AS first_run_start_time,
  (SELECT MAX(s.start_time) FROM scoped s WHERE s.benchmark_id = bench.benchmark_id) AS last_run_start_time,
  (SELECT COUNT(DISTINCT s.hardware_sku) FROM scoped s WHERE s.benchmark_id = bench.benchmark_id) AS hardware_sku_count,
  (SELECT COUNT(DISTINCT s.host_id)      FROM scoped s WHERE s.benchmark_id = bench.benchmark_id) AS host_count,
  (SELECT COUNT(*) FROM task_results t JOIN scoped s ON s.run_id = t.run_id
     WHERE s.benchmark_id = bench.benchmark_id)                                     AS task_attempts,
  (SELECT COUNT(*) FROM task_results t JOIN scoped s ON s.run_id = t.run_id
     WHERE s.benchmark_id = bench.benchmark_id AND t.passed = 1)                    AS task_attempts_passed,
  (SELECT CASE WHEN COUNT(*) > 0 THEN 1.0 * SUM(CASE WHEN t.passed = 1 THEN 1 ELSE 0 END) / COUNT(*) END
     FROM task_results t JOIN scoped s ON s.run_id = t.run_id
     WHERE s.benchmark_id = bench.benchmark_id)                                     AS pass_rate,
  (SELECT COUNT(DISTINCT COALESCE(t.logical_task_key, s.benchmark_id || '::' || t.task_id))
     FROM task_results t JOIN scoped s ON s.run_id = t.run_id
     WHERE s.benchmark_id = bench.benchmark_id)                                     AS distinct_logical_tasks,
  (SELECT COUNT(t.duration_s) FROM task_results t JOIN scoped s ON s.run_id = t.run_id
     WHERE s.benchmark_id = bench.benchmark_id)                                     AS duration_sample_n,
  (SELECT AVG(t.duration_s) FROM task_results t JOIN scoped s ON s.run_id = t.run_id
     WHERE s.benchmark_id = bench.benchmark_id)                                     AS task_duration_s_avg,
  (SELECT MIN(t.duration_s) FROM task_results t JOIN scoped s ON s.run_id = t.run_id
     WHERE s.benchmark_id = bench.benchmark_id)                                     AS task_duration_s_min,
  (SELECT MAX(t.duration_s) FROM task_results t JOIN scoped s ON s.run_id = t.run_id
     WHERE s.benchmark_id = bench.benchmark_id)                                     AS task_duration_s_max,
  (SELECT COUNT(*)          FROM spans sp JOIN scoped s ON s.run_id = sp.run_id
     WHERE s.benchmark_id = bench.benchmark_id AND sp.span_kind = 'llm_call')       AS llm_call_count,
  (SELECT SUM(sp.tokens_in) FROM spans sp JOIN scoped s ON s.run_id = sp.run_id
     WHERE s.benchmark_id = bench.benchmark_id AND sp.span_kind = 'llm_call')       AS llm_tokens_in_total,
  (SELECT SUM(sp.tokens_out) FROM spans sp JOIN scoped s ON s.run_id = sp.run_id
     WHERE s.benchmark_id = bench.benchmark_id AND sp.span_kind = 'llm_call')       AS llm_tokens_out_total,
  (SELECT SUM(sp.cost_usd)  FROM spans sp JOIN scoped s ON s.run_id = sp.run_id
     WHERE s.benchmark_id = bench.benchmark_id AND sp.span_kind = 'llm_call')       AS llm_cost_usd_total,
  -- Hardware counters are published PER LAYER, never blended: l3 (perf-stat) and
  -- perfspect are two different instruments and their ipc means differ by ~2x on
  -- the same workload. Each mean ships with its own sample count.
  (SELECT COUNT(*)   FROM measurements m JOIN scoped s ON s.run_id = m.run_id
     WHERE s.benchmark_id = bench.benchmark_id
       AND m.layer = 'l3' AND m.ipc IS NOT NULL)                                    AS ipc_sample_rows_l3,
  (SELECT AVG(m.ipc) FROM measurements m JOIN scoped s ON s.run_id = m.run_id
     WHERE s.benchmark_id = bench.benchmark_id
       AND m.layer = 'l3' AND m.ipc IS NOT NULL)                                    AS ipc_avg_l3,
  (SELECT COUNT(*)   FROM measurements m JOIN scoped s ON s.run_id = m.run_id
     WHERE s.benchmark_id = bench.benchmark_id
       AND m.layer = 'perfspect' AND m.ipc IS NOT NULL)                             AS ipc_sample_rows_perfspect,
  (SELECT AVG(m.ipc) FROM measurements m JOIN scoped s ON s.run_id = m.run_id
     WHERE s.benchmark_id = bench.benchmark_id
       AND m.layer = 'perfspect' AND m.ipc IS NOT NULL)                             AS ipc_avg_perfspect,
  -- cache_miss_pct is l3-only in practice (perfspect payloads carry no such key),
  -- so the layer predicate here documents that rather than restricting anything.
  (SELECT COUNT(*)              FROM measurements m JOIN scoped s ON s.run_id = m.run_id
     WHERE s.benchmark_id = bench.benchmark_id
       AND m.layer = 'l3' AND m.cache_miss_pct IS NOT NULL)                         AS cache_miss_pct_sample_rows_l3,
  (SELECT AVG(m.cache_miss_pct) FROM measurements m JOIN scoped s ON s.run_id = m.run_id
     WHERE s.benchmark_id = bench.benchmark_id
       AND m.layer = 'l3' AND m.cache_miss_pct IS NOT NULL)                         AS cache_miss_pct_avg_l3,
  (SELECT group_concat(l.layer) FROM (SELECT DISTINCT m.layer FROM measurements m
      JOIN scoped s ON s.run_id = m.run_id
      WHERE s.benchmark_id = bench.benchmark_id ORDER BY m.layer) l)                AS measurement_layers
FROM bench
LEFT JOIN benchmarks b ON b.benchmark_id = bench.benchmark_id;
```

`pass_rate` is computed from `task_results`, **never** from the denormalized
`runs.passed_tasks`/`total_tasks` — those are populated and currently agree, but they
are unenforced denormalizations that can drift, so the view derives the number it
publishes.

**The hardware counters are per-layer on purpose.** An earlier revision of this view
published a single `ipc_avg` guarded only on `m.ipc IS NOT NULL`, which silently
averaged `l3` and `perfspect` together: on the live store that produced
`synthetic-cpu ipc_avg=1.897` over 34 rows, blending `l3` (n=22, 2.373) with
`perfspect` (n=12, 1.024) — two different instruments in one published number, and
`ipc` is a leaderboard column. `ipc_avg_l3`/`ipc_avg_perfspect` each carry their own
sample count so a surface can suppress a mean built from too few rows; never re-blend
them. `cache_miss_pct_avg_l3` names its layer for the same reason — it looks like a
counterpart to `ipc` but perfspect never emits the key (§3.3).
`hardware_sku_count > 1` or `host_count > 1` means the row **mixes machines** and must
not be presented as a single-platform number.

#### 7.3.7 `v_logical_task_summary` — per-logical-task comparison across runs

*Grain:* one row per `logical_task_key`. **16 columns.** *Verified:* executes on the
live DB (10 rows as of 2026-07-30) — e.g. `synthetic-cpu::compile` at `attempts=8,
run_count=8, pass_rate=1.0, duration_sample_n=8`, and single-attempt keys correctly
reporting `p50 = p95 =` the one value.

```sql
CREATE TEMP VIEW v_logical_task_summary AS
WITH scoped AS (
  SELECT r.run_id, r.benchmark_id, r.hardware_sku
  FROM runs r
  WHERE r.benchmark_id IS NOT NULL
    AND NOT EXISTS (SELECT 1 FROM sweeps sw       WHERE sw.sweep_id = r.run_id)
    AND NOT EXISTS (SELECT 1 FROM sweep_points sp WHERE sp.run_id  = r.run_id)
    AND COALESCE(CASE WHEN json_valid(r.metadata)
                      THEN json_extract(r.metadata, '$.run_kind') END, '')
        NOT IN ('sweep', 'sweep_cell')
),
att AS (
  SELECT COALESCE(t.logical_task_key, s.benchmark_id || '::' || t.task_id) AS agg_key,
         t.run_id, t.task_id, t.passed, t.duration_s, t.num_turns, s.hardware_sku
  FROM task_results t
  JOIN scoped s ON s.run_id = t.run_id
),
ranked AS (
  SELECT agg_key, duration_s,
         ROW_NUMBER() OVER (PARTITION BY agg_key ORDER BY duration_s) AS rn,
         COUNT(*)     OVER (PARTITION BY agg_key)                     AS n
  FROM att WHERE duration_s IS NOT NULL
),
pct AS (
  SELECT agg_key,
         -- nearest-rank p50: rank ceil(n/2). Exact for odd n; for even n it is the
         -- LOWER of the two central values and never interpolates.
         MIN(CASE WHEN rn >= n / 2.0  THEN duration_s END) AS duration_s_p50,
         MIN(CASE WHEN rn >= 0.95 * n THEN duration_s END) AS duration_s_p95,
         COUNT(*)                                               AS duration_sample_n
  FROM ranked GROUP BY agg_key
)
SELECT
  a.agg_key                                        AS logical_task_key,
  substr(a.agg_key, 1, instr(a.agg_key, '::') - 1) AS benchmark_id,
  substr(a.agg_key, instr(a.agg_key, '::') + 2)    AS task_id,
  COUNT(*)                                         AS attempts,
  COUNT(DISTINCT a.run_id)                         AS run_count,
  SUM(CASE WHEN a.passed = 1 THEN 1 ELSE 0 END)    AS attempts_passed,
  1.0 * SUM(CASE WHEN a.passed = 1 THEN 1 ELSE 0 END) / COUNT(*) AS pass_rate,
  COUNT(DISTINCT a.hardware_sku)                   AS hardware_sku_count,
  MIN(a.duration_s)                                AS duration_s_min,
  AVG(a.duration_s)                                AS duration_s_avg,
  MAX(a.duration_s)                                AS duration_s_max,
  MAX(p.duration_s_p50)                            AS duration_s_p50,
  MAX(p.duration_s_p95)                            AS duration_s_p95,
  COALESCE(MAX(p.duration_sample_n), 0)            AS duration_sample_n,
  SUM(CASE WHEN a.duration_s IS NULL THEN 1 ELSE 0 END) AS attempts_duration_null,
  AVG(a.num_turns)                                 AS num_turns_avg
FROM att a
LEFT JOIN pct p ON p.agg_key = a.agg_key
WHERE a.agg_key IS NOT NULL
GROUP BY a.agg_key;
```

NULL stored keys are **repaired** inside the CTE rather than dropped, so pre-0003
migrated rows are not silently lost — a bare `GROUP BY logical_task_key` would lose
them.

SQLite has no percentile function, so p50/p95 are **nearest-rank order statistics**
computed with `ROW_NUMBER`/`COUNT` window functions over the non-NULL durations.
Read that literally: `duration_s_p50` is the value at rank `ceil(n/2)`, so for even
`n` it is the **lower** of the two central values and is **not** the interpolated
median — n=2 `[10,20]` → 10.0 (median 15.0); n=4 `[1,2,3,4]` → 2.0 (median 2.5). For
odd `n` it is exact. If you need a true median, compute it yourself from the
attempts. **Publish `duration_sample_n` next to both percentiles** — a p95 over 3
samples is just the max. `attempts_duration_null` counts attempts excluded from the
latency stats (NULL duration, not zero duration).

#### 7.3.8 `v_sweep_density` — the scaling curve

*Grain:* one row per (sweep_id, density) — replicates averaged. **32 columns**, in
the SELECT order below (note the tail is `node_ctx_sw_per_s_avg`,
`node_mem_avail_mb_min`, `node_iowait_pct_avg` — project by name, never by ordinal).
*Verified:* 0 rows on the live DB (no sweep has ever been run against the canonical
store — the only view still genuinely empty there); 4 rows against a purpose-built
4-density × 2-replicate sweep, where `density=1.0`
returned `concurrency_requested=64, cell_count=2, replicate_count=2,
throughput_per_min_avg=12.0, runqueue_per_vcpu=0.1406, data_source='measured',
cells_missing_run_row=0`, and `density=2.0` correctly flagged
`cells_p95_latency_zero=2`.

```sql
CREATE TEMP VIEW v_sweep_density AS
SELECT
  sp.sweep_id,
  s.benchmark                                      AS sweep_benchmark,
  s.created_at                                     AS sweep_created_at,
  s.hardware_sku,
  s.vcpu_basis,
  s.vcpu_basis_kind,
  s.numa_policy,
  s.model,
  s.replay_fixture,
  CASE WHEN json_valid(s.metadata) THEN json_extract(s.metadata, '$.data_source') END AS data_source,
  sp.density,
  MAX(sp.concurrency)                              AS concurrency_requested,
  COUNT(*)                                         AS cell_count,
  COUNT(DISTINCT sp.replicate)                     AS replicate_count,
  group_concat(sp.run_id)                          AS cell_run_ids,
  SUM(CASE WHEN EXISTS (SELECT 1 FROM runs r WHERE r.run_id = sp.run_id) THEN 0 ELSE 1 END) AS cells_missing_run_row,
  AVG(sp.throughput_per_min)                       AS throughput_per_min_avg,
  MIN(sp.throughput_per_min)                       AS throughput_per_min_min,
  MAX(sp.throughput_per_min)                       AS throughput_per_min_max,
  AVG(sp.elapsed_s)                                AS elapsed_s_avg,
  SUM(sp.completed_trials)                         AS completed_trials_sum,
  SUM(CASE WHEN sp.completed_trials IS NULL THEN 1 ELSE 0 END) AS cells_completed_trials_null,
  MAX(sp.p95_trial_latency_s)                      AS p95_trial_latency_s_max,
  SUM(CASE WHEN sp.p95_trial_latency_s = 0.0 THEN 1 ELSE 0 END) AS cells_p95_latency_zero,
  AVG(sp.cpu_avg)                                  AS node_cpu_pct_avg,
  MAX(sp.cpu_p95)                                  AS node_cpu_pct_p95_max,
  MAX(sp.cpu_peak)                                 AS node_cpu_pct_peak_max,
  MAX(sp.runqueue_max)                             AS node_runqueue_max,
  CASE WHEN s.vcpu_basis > 0 THEN MAX(sp.runqueue_max) * 1.0 / s.vcpu_basis END AS runqueue_per_vcpu,
  AVG(sp.ctx_sw_per_s)                             AS node_ctx_sw_per_s_avg,
  MIN(sp.mem_avail_mb_min)                         AS node_mem_avail_mb_min,
  AVG(sp.iowait_pct_avg)                           AS node_iowait_pct_avg
FROM sweep_points sp
JOIN sweeps s ON s.sweep_id = sp.sweep_id
GROUP BY sp.sweep_id, sp.density;
```

Percentile and extremum columns are deliberately `MAX`/`MIN`, never `AVG`, because
averaging a p95 or a peak across replicates is invalid. `runqueue_per_vcpu` is the
saturation signal (≫ 1 = oversubscribed). Three integrity counters exist because the
sweep tier has three documented defects: `cells_p95_latency_zero` (p95 persisted as
0.0), `cells_missing_run_row` (no FK on `sweep_points.run_id`), and
`cells_completed_trials_null` (NULL ≠ 0). **Cross-sweep comparison additionally
requires equal `vcpu_basis`, `vcpu_basis_kind` and `numa_policy`** — and note
`numa_policy` is currently stored but not applied.

### 7.4 Worked example — an individual run page

```sql
-- 1. Header. Which panels have data is answered by the coverage columns,
--    so the page never renders an empty chart as "0".
SELECT run_id, benchmark_id, benchmark_display_name, start_time, wall_s, status,
       hardware_sku, host_id, model, optimization_profile, numa_policy,
       agentsysperf_version,
       tasks_observed, tasks_passed_observed, pass_rate_observed,
       total_tasks_reported, task_count_agrees,
       measurement_rows, measurement_layers, span_rows, verdict_rows,
       artifact_count, artifacts_json
FROM v_run_header
WHERE run_id = :run_id;

-- 2. Task table.
SELECT task_id, logical_task_key, passed, duration_s, num_turns,
       measurement_rows_matched, span_rows_matched
FROM v_run_task
WHERE run_id = :run_id
ORDER BY task_id;

-- 3. Process-level resource metrics. Note the explicit layer filter: without it
--    the l1_system rows contribute NULLs and the l3 rows contribute nothing.
SELECT span_id, derived_task_id, duration_s_effective, cpu_time_s,
       proc_cpu_pct_mean, proc_cpu_pct_peak, rss_kb_peak, proc_threads_peak
FROM v_run_measurement
WHERE run_id = :run_id AND layer = 'l1'
ORDER BY seq;

-- 4. Hardware counters. Same discipline, different layer -- and note the layer
--    column is SELECTed, because l3 and perfspect are different instruments whose
--    ipc must not be averaged together. cache_miss_pct/llc_miss_per_s/
--    branch_miss_pct are NULL on every perfspect row (l3-only keys).
SELECT span_id, derived_task_id, layer,
       ipc, cache_miss_pct, llc_miss_per_s, branch_miss_pct
FROM v_run_measurement
WHERE run_id = :run_id AND layer IN ('l3','perfspect') AND ipc IS NOT NULL
ORDER BY layer, seq;

-- 5. Node saturation over the run (l1_system data lives only in payload JSON).
SELECT span_id, node_cpu_pct_avg, node_cpu_pct_p95, node_cpu_pct_peak,
       node_runqueue_max, node_ctx_sw_per_s_avg, node_iowait_pct_avg,
       node_mem_avail_mb_min, node_logical_cpus
FROM v_run_measurement
WHERE run_id = :run_id AND layer = 'l1_system'
ORDER BY seq;

-- 6. Verdicts, separated by scope so 'unknown' is never shown as a task name.
SELECT analyzer_name, verdict, confidence, verdict_scope, verdict_scope_id,
       evidence_json, recommendations_json, recommendation_count
FROM v_run_verdict
WHERE run_id = :run_id
ORDER BY verdict_scope, analyzer_name;

-- 7. Step decomposition. Empty for CLI/synthetic runs; render "no step trace",
--    not "$0.00 / 0 tokens".
SELECT span_task_id, span_kind, span_count, ok_count, error_count,
       duration_us_sum, duration_us_zero_count,
       llm_tokens_in, llm_tokens_out, llm_cost_usd, model_ids, tool_names
FROM v_run_span_rollup
WHERE run_id = :run_id
ORDER BY span_task_id, span_kind;
```

### 7.5 Worked example — an aggregated leaderboard page

```sql
-- Benchmark leaderboard. Order by pass_rate but gate on sample size, and expose
-- the mixing flags so a row spanning two hosts is never sold as one platform.
SELECT benchmark_id,
       COALESCE(display_name, benchmark_id) AS label,
       is_registered,
       run_count, task_attempts, pass_rate,
       distinct_logical_tasks,
       task_duration_s_avg, duration_sample_n,
       ipc_avg_l3, ipc_sample_rows_l3,
       ipc_avg_perfspect, ipc_sample_rows_perfspect,
       cache_miss_pct_avg_l3, cache_miss_pct_sample_rows_l3,
       llm_call_count, llm_tokens_in_total, llm_cost_usd_total,
       hardware_sku_count, host_count,
       measurement_layers,
       last_run_start_time
FROM v_benchmark_summary
WHERE task_attempts > 0
ORDER BY pass_rate DESC, run_count DESC;

-- Per-task detail for one benchmark. p50/p95 come with their sample count so the
-- surface can grey out a percentile computed from 2 attempts.
SELECT task_id, attempts, run_count, pass_rate,
       duration_s_p50, duration_s_p95, duration_sample_n,
       duration_s_min, duration_s_max, attempts_duration_null,
       hardware_sku_count
FROM v_logical_task_summary
WHERE benchmark_id = :benchmark_id
ORDER BY duration_s_p95 DESC NULLS LAST;

-- Regression watch: has a task's latency moved between the two most recent runs?
WITH recent AS (
  SELECT run_id, start_time,
         ROW_NUMBER() OVER (ORDER BY start_time DESC, run_id DESC) AS rn
  FROM v_run_header
  WHERE benchmark_id = :benchmark_id
)
SELECT t.agg_key,
       MAX(CASE WHEN r.rn = 1 THEN t.duration_s END) AS latest_s,
       MAX(CASE WHEN r.rn = 2 THEN t.duration_s END) AS previous_s,
       MAX(CASE WHEN r.rn = 1 THEN t.passed END)     AS latest_passed
FROM v_run_task t
JOIN recent r ON r.run_id = t.run_id
WHERE r.rn <= 2
GROUP BY t.agg_key
ORDER BY t.agg_key;

-- Scaling curve for one sweep, with the interpretation context and the integrity
-- flags on every row. Refuse to plot if data_source <> 'measured'.
SELECT density, concurrency_requested, replicate_count,
       throughput_per_min_avg, throughput_per_min_min, throughput_per_min_max,
       p95_trial_latency_s_max, cells_p95_latency_zero,
       node_cpu_pct_avg, node_runqueue_max, runqueue_per_vcpu,
       vcpu_basis, vcpu_basis_kind, numa_policy, data_source,
       cells_missing_run_row, cells_completed_trials_null
FROM v_sweep_density
WHERE sweep_id = :sweep_id
ORDER BY density;
```

Note `ORDER BY start_time DESC, run_id DESC` in the regression query — **not**
`ORDER BY run_id`, which is not chronological, and not `MAX(run_id)`.

### 7.6 Pitfalls

**Units, written down once.**

| Unit | Columns |
|---|---|
| seconds (epoch) | `runs.start_time`/`end_time`, `benchmarks.created_at`, `sweeps.created_at` |
| seconds (duration) | `task_results.duration_s`, `sweep_points.elapsed_s`, `p95_trial_latency_s`, `measurements.cpu_time_s`, payload `duration_s` |
| microseconds | `measurements.duration_us`, `spans.start_ts_us`/`end_ts_us`/`duration_us` |
| kilobytes | `measurements.rss_kb_peak` |
| megabytes | `sweep_points.mem_avail_mb_min` |
| percent | `cpu_pct_mean`, `cache_miss_pct`, `cpu_avg`/`p95`/`peak`, `iowait_pct_avg`, `branch_miss_pct` |
| per second | `llc_miss_per_s`, `ctx_sw_per_s` (actually a windowed **average**) |
| per minute | `sweep_points.throughput_per_min` |

**NULL versus zero, and the schema is internally inconsistent about it.**
`measurements` is NULL-disciplined by design (0002's comment: *"NULL != 0 — a
missing key stays NULL, never 0"*, pinned by `test_p1_null_not_zero`), so `0.0`
means the probe measured zero. `spans` is the **opposite** — defaults 0/0/0/0.0/'ok'
at both DDL and StepTrace layers, so an uncosted LLM call reads as free and an unset
outcome reads as success. `runs` is a third case: three of its DDL defaults
(`hardware_sku`, `total_tasks`, `passed_tasks`) are **dead** because
`store_run_metadata` binds `metadata.get(...)` for each, so an omitted key stores NULL
rather than the default. That does **not** mean the columns are empty — real callers
pass the counters (`run_driver.py:344`), and on the live store all 10 runs have
`total_tasks`/`passed_tasks` populated and agreeing with `task_results`
(`v_run_header.task_count_agrees = 1` for every run). Show the discrepancy columns;
just derive published rates from `task_results`, since nothing enforces the agreement.
`sweep_points` is a fourth: `completed_trials` NULL means never measured while 0
means everything failed. Two tables in one schema with contradictory missing-data
philosophies — never write a generic "treat null as 0" coercion.

**Empty-but-valid is the common case.** On the live store `sweeps`, `sweep_points`,
`benchmarks` and `artifacts` are 0 rows, and `spans` is populated for exactly one run
out of ten. The cause is structural, not transient: `RunContext.stop()` writes only
`runs` + `measurements` + `spans`; `run_driver` adds `task_results` + verdicts in a
**second** `persist_run`; `live_dashboard` writes nothing at all; `store_spans` is
wired only on the TB2/agent-loop path. So a page assuming "every run has
task_results", or "spans is either empty or complete", will be wrong. Render "no
data", not zero, and decide **per run** from `v_run_header`'s coverage columns.

**A run can exist as JSON and not in the DB.** `persist_run` failure inside
`RunContext.stop()` is caught and downgraded to a log warning *after*
`measurement_records.json` has already been written. `agentsysperf analyze` reads
only the JSON. The two views can honestly disagree.

**Do not INNER JOIN on any derived `task_id`.** Three incompatible rules share the
name (§4). Use the count bridges and `verdict_scope` in the views.

**Do not INNER JOIN `measurements` to `spans`.** There is no FK, and on the live DB
9 of 10 runs have measurements and no spans at all — the join silently drops them.

**Sweep and placeholder rows pollute naive run lists.** Exclude them three ways
(existence in `sweeps`, existence in `sweep_points`, `metadata.run_kind`), plus
`metadata.backfilled_orphan` — migrations 0003/0005 insert placeholder runs with
`start_time = 0`. **The eight views do NOT all do this**, so know which you are
reading:

| View | Sweep/cell exclusion | `backfilled_orphan` |
|---|---|---|
| `v_run_header` | yes (all three) | **yes** |
| `v_run_task` | yes (all three) | **yes** |
| `v_benchmark_summary`, `v_logical_task_summary` | yes (all three, in their `scoped` CTE) | no — placeholder runs have no `task_results`, so they contribute nothing |
| `v_run_measurement`, `v_run_verdict`, `v_run_span_rollup` | **none** — bare `FROM measurements`/`analyzer_verdicts`/`spans` | no |

Reproduced on a purpose-built store with one sweep: after writing one measurement,
one span and one verdict against the sweep cell `sw1::d1_r0`, `v_run_measurement`,
`v_run_span_rollup` and `v_run_verdict` all returned those rows while `v_run_header`
and `v_run_task` correctly excluded them. **Join the three unscoped views to
`v_run_header` (or filter them yourself) before aggregating**, or a page
double-counts sweep-cell rows as ordinary runs.

**`SELECT *` is not a contract.** `list_runs`, `get_run`, `query_spans`,
`query_sweeps` and `query_sweep_points` all `SELECT *`. 0004 added nine keys to the
`runs` row and 0007 renamed one; 0003 reordered `task_results`. Project named
columns; never index positionally; never enumerate keys.

**Prometheus/Grafana is a lossy secondary view.** The exporter branches on only
`l1`/`l3`/`perfspect`, silently dropping `l1_system` and `emon`; labels derive
`task_id` by the split rule; retention is 30 days. Its aggregates **will** disagree
with SQL over this DB. Query the DB (or the DuckDB attach) for anything published.

**Hardcoded `/tmp` paths are dead ends.** `/tmp/agentsysperf_results/…`,
`/tmp/agentsysperf_live/run_<ts>`, `/tmp/agentsysperf_scaling_mixed/all_results.json`
are probed by the existing dashboards and are almost all absent, so those panels
render empty rather than erroring. A new consumer should read the store exclusively.

**Provenance obligations before publishing a number.** Surface
`sweeps.metadata.data_source` — `'synthetic'` means **modeled** dry-run points.
Surface `cells_p95_latency_zero` and treat `concurrency` as requested, not achieved.
Treat `sweeps.numa_policy` as unenforced. `runs.agentsysperf_version` and
`result_digest` are NULL on every real run, so **no shipped run is currently
reproducible-by-record** — say so rather than omitting the field.
`sweeps.model = 'agentsysperf-proxy'` means the LLM was replayed.

**Snapshot with `VACUUM INTO`, never `cp`** (§6).

---

## 8. Known gaps and planned work

### 8.1 SHIPPED but wrong / drifting

| Gap | Impact | Where |
|---|---|---|
| **Dead column defaults on `runs`.** `total_tasks` and `passed_tasks` declare `DEFAULT 0` that can never fire, because `store_run_metadata` always binds an explicit NULL. (`hardware_sku`'s fabricated default was removed by migration 0008.) | A consumer reading the DDL expects `0`; it gets NULL. Do not treat NULL as zero — see §7.6. | `schema.sql`, `store_run_metadata` |
| **Migration runner is not atomic across the FK gate, and its error message lies.** Commit happens before `foreign_key_check`. | A failed migration can leave the DB versioned as successful with bad data while telling the operator the opposite; the next open skips it. | `_run_migrations` |
| **Migration numbering has no collision detection.** `0007` has two file claimants (committed `0007_rename_agentsysperf_version.sql` vs PR #5's `0007_agent_side_metric.sql`), plus a third unnumbered proposal. | `sorted()` applies whichever sorts first and silently skips the other. Reproduced with PR #5's files in place: an **existing** v7 DB fails to OPEN — `RuntimeError: Migration 0008_rename_agentperf_version_column.sql failed (DB left at version 7): no such column: "agentperf_version"` — while a **fresh** DB lands at v8 with the right shape. Same migration set, opposite outcomes. | main, PR #5 (`0007` + `0008`) |
| **`_create_fallback_schema` creates only `runs`, `task_results`, `analyzer_verdicts`, `spans`** — five of the nine tables are absent from it. | `measurements`/`benchmarks`/`artifacts` are recovered only because migrations 0002/0004/0006 then run on top; `sweeps` and `sweep_points` appear in **no** migration and are therefore *permanently* absent on a fallback DB. Reproduced by moving `schema.sql` aside: `delete_run('r1')` logs *delete_run failed for r1: no such table: sweep_points*, returns 0, and leaves the row. Every sweep read/write also fails. | `sqlite_store.py` |
| **`schema.sql` is now a misleading document of the effective schema.** Its `task_results`/`analyzer_verdicts` bodies were never updated after 0003/0005/0007. | Reading it shows a global `task_id` PK, no `logical_task_key`, no verdict UNIQUE, no cascade FKs. Three sources of truth that only agree after the runner finishes. | `schema.sql` |
| **`store_spans` silently drops 12 StepTrace fields**, including `routing_reason` and `kv_cache_hit`, while both sides claim `schema_version '0.3'`. | Persisting spans is lossy with no error and no warning; the "byte-compatible with AgentOptimizer v0.3" claim does not survive a round-trip. | `store_spans` |
| **`spans` upsert omits `schema_version` from its `DO UPDATE SET` list** (16 of the 17 non-key columns are updated). | Re-storing a span from a newer StepTrace overwrites every value but keeps the old version tag. | `store_spans` |
| **`analyzer_verdicts`' `'unknown'` sentinel collides under the 0005 UNIQUE key.** | All run-level verdicts from one analyzer in one run upsert over each other; only the last survives. 0005 fixed duplication and introduced this loss; no test covers it. | `store_analysis_results` |
| **`query_verdicts` decodes `evidence` unguarded**, and the enclosing `except sqlite3.Error` does not catch `JSONDecodeError`. | A malformed blob crashes a report or dashboard, unlike every other decode site in the store. | `query_verdicts` |
| **`measurements` shipped shape diverges from its own design doc**: no `node_id`/`kind` columns, INTEGER not REAL for two metrics, plus a `seq` column and a UNIQUE constraint the doc never mentions. | Code written against the plan doc looks for `measurements.node_id` and finds nothing. | `STORAGE_UNIFICATION_PLAN.md` lines 53–68 vs `0002` |
| **`measurements.layer` DDL comment omits `l1_system`**, which is a third of real rows, is `ScalingAnalyzer.input_layers`, and populates **zero** promoted columns. | Averaging `cpu_pct_mean` over a run silently sees only `l1` rows and none of the node-level data. | `0002` |
| **Unit inconsistency across tables and within `payload`** (seconds vs microseconds, KB vs MB, `duration_us` vs `duration_s` by layer). | Silent 1e6 errors. | §7.6 |
| **Five unrelated "version" concepts** (`measurements.schema_version` `'0.4'`, `spans.schema_version` `'0.3'`, `user_version` 7, `runs.agentsysperf_version`, `benchmarks.version`). | A consumer checking "schema_version" must know which table it came from. | — |
| **Commit-discipline split**: six writers honour `persist_run`'s transaction, six bypass it with direct `conn.commit()`. | Latent today; adding such a call inside `persist_run` would break atomicity with no test catching it. | §6 |
| **`benchmarks.benchmark_id` is not an FK target, and the plan doc says it is.** | `SELECT FROM benchmarks` returns nothing while every run carries a slug; `delete_benchmark` is app-level. | `STORAGE_UNIFICATION_PLAN.md` line 85 vs `0004` |
| **`sweep_points.run_id` has no FK at all**, and the sweep_id/run_id namespace is overloaded three ways. | A point can reference a nonexistent run silently; any naive run count includes synthetic placeholders. | §4 |
| **`artifacts.created_at` and `metadata` are unwritable** (hardcoded 0 and NULL), and 3 of the 4 documented `kind` values are never written. | 0006's stated purpose — never globbing `/tmp` — is unrealized for scaling plots and analysis reports, which `demo_app` still discovers by glob. | `store_artifact` |
| **Read-only stores never migrate, silently.** | A viewer on a stale file sees the old shape with no error and no warning. Feature-detect. | §6, §7.2 |
| **`sweep_points.p95_trial_latency_s` is persisted as 0.0 on every real sweep**, dropping `ScalingAnalyzer` confidence 0.8 → 0.6 and flat-lining the p95 trace while the knee verdict string stays identical. | Nobody notices. | internal UX assessment |
| **`sweeps.numa_policy` is stored but never applied** — `socket_pinned` and `unpinned` sweeps ran byte-identical. | Any NUMA-labelled view is misleading; prior `socket_pinned` data must be invalidated before shipping a NUMA comparison. | internal scaling-viz review |
| **Sweep density does not actually vary concurrency** (semaphore ceiling over `n_attempts × n_tasks`). | A "knee" derived from six copies of one operating point. | internal UX assessment |
| **Minor.** DuckDB `export_parquet` f-string-interpolates the table name and output path into SQL with no allowlist. `store_spans` logs `len(list(spans))` after iterating (a generator reads 0). `get_artifact_path` orders DESC while `list_artifacts` orders ASC. `schema.sql`'s header still says "AgentPerf". `get_run` returns None for both "no such run" and "query failed". `sweeps.benchmark` is named inconsistently and has no FK. `PRAGMA journal_mode=WAL` on a read-only handle succeeds as a no-op, so it is not a writability probe (only a mode *change* errors, with `disk I/O error`). `PRAGMA query_only=1` blocks `CREATE TEMP VIEW`, so the §7.3 views must be installed before it is set (§7.2). `duckdb_analytics.ipc_by_benchmark`'s `avg(cache_miss_pct)` is silently `l3`-only while its `ipc` gate admits perfspect rows too. `README.md` line 19 lists 8 table names, omitting `sweep_points`. | | — |

### 8.2 PLANNED — not shipped

| Item | Status | Source |
|---|---|---|
| ULID `run_id`s + `runs.run_label` | **PLANNED.** Shipped ids are `{benchmark}_{stamp}` or `run-{uuid8}`, so `ORDER BY run_id` is not chronological. | `STORAGE_UNIFICATION_PLAN.md` §1 |
| Soft-delete tombstones (`status='deleted'`, `deleted_at`, `reason`, retained `result_digest`) | **PLANNED.** Hard delete + cascade shipped; the withdrawn-result half depends on a deferred leaderboard. | §3 |
| `measurements.node_id` / `kind` as columns | **PLANNED / superseded.** They live only as payload keys. | lines 59–60 |
| JSON1 expression indexes over `payload` | **PLANNED.** Zero exist; unpromoted-key filters are full scans. | line 246 |
| Postgres backend as an entry-point plugin | **SEAM ONLY, deliberately.** Any non-`sqlite` scheme raises `NotImplementedError`. Trigger-gated. | §4/§8, `STORAGE_IMPLEMENTATION_PLAN.md` P4 |
| `tar.zst` export/import bundle with manifest + digest validation | **PARTIAL.** Parquet export shipped; the bundle format, manifest and validating `import` did not. | §8 |
| `ArtifactStore` (local \| s3) mini-protocol | **PLANNED (later tier).** The `artifacts` table shipped; the pluggable backend did not. | §8 |
| `sweeps`/`sweep_points` widening (cpu_vendor, microarch, logical_cpus, numa_nodes, price_usd_per_hr, node_label, engine, numa_topology; placement/phase_mix) | **PLANNED.** Needs renumbering off `0007`. | internal scaling-viz review |
| `sweep_points` written from the new concurrency path | **PLANNED, 2.0 d, self-labelled unverified.** The table's shape is a de-facto stable contract as long as the replacement keeps writing here. | internal UX assessment |
| Run env-provenance capture (kernel, microcode, governor, `perf_event_paranoid`, hugepages, dep-lock digest, git SHA) wired into run metadata | **PLANNED, v1-blocking.** The provenance *columns* exist (0004) but are not populated from the detected environment. | internal release plan §4 |
| `agentsysperf serve` / `migrate` subcommands | **NOT SHIPPED.** Migration ships as `scripts/migrate_to_unified_store.py`. | `STORAGE_IMPLEMENTATION_PLAN.md` P9 |
| `agentsysperf results save` / `compare` + the `results/{onprem,cloud}/…` tree with mandatory `platform.json` | **PLANNED, unstarted.** | internal results-directory plan |
| Store-only flip (`start.sh`/exporter driven from the store, DAL fallbacks removed) | **NOT SHIPPED, by design and by gate** — gated on `verify_backfill_parity.py`. The legacy JSON/`/tmp` read paths remain as a verbatim fallback. | `STORAGE_IMPLEMENTATION_PLAN.md` P7/P10 |
| Retention / compaction policy; the hot/cold cutline | **OPEN QUESTION**, never decided. | §8b "Caveats / open" |
| Measured DuckDB-vs-SQLite crossover on real data | **NOT DONE.** The ~10–15k-row figure is asserted, never measured; the plan itself rates the claim medium confidence. | §8b Caveats |
| OTel emission from `track_span` (gen_ai.* + cost + run_id) → Langfuse + Tempo | **PLANNED.** Phases 0–2 done; Phase 3 unshipped. `gen_ai.*` is experimental and OTel defines no cost attribute. | internal trace-analytics plan |
| L2 flame graphs | **NO-GO as specified, deferred.** Three independently fatal reasons. | `phase_metrics_and_L2_plan.md` Feature B |
| L4 (scheduler/SMT/GIL), L5 (socket/uncore via intel_pcm) measurement layers | L4 **PLANNED**; L5 **externally blocked** on Intel PMU event codes for EMR. Do not expect an uncore layer. | `measurement_layers.md` |
| NetworkTelemetry (Protocol 6), Recommender (Protocol 7), public HTML report / MLPerf-style leaderboard, CSV + OTEL-JSON export | **DEFERRED post-v1.0.** No storage tables specified. | internal release plan §6 |

### 8.3 The standing rule

> "Never infer a column from a planning doc — planning docs describe intent and
> frequently diverge from what shipped. When a doc and the code disagree, **the code
> wins** and you flag the drift explicitly."
> — `.claude/agents/database-expert.md`

That rule produced §8.1. Apply it to this document too: before depending on anything
here, re-verify against `src/storage/schema.sql` plus
`src/storage/migrations/*.sql`, and treat `PRAGMA user_version` as
necessary but not sufficient (§5).
