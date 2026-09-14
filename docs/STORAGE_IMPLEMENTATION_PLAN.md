# AgentSysPerf Storage — Execution Plan (dashboard-safe, verifier-corrected)

Companion to `STORAGE_UNIFICATION_PLAN.md` (the *what/why*). This is the *how*:
an ordered, independently-shippable phase plan that keeps all four dashboards
(demo :7860, live :7861, grafana :3000, langfuse :7862) working throughout.

Produced by a map → design → adversarial-verify workflow. The verifiers returned
**UNSAFE for live_dashboard** and SAFE-WITH-GAPS for the other three on the first
draft; the **16 fixes below are folded in**. Decisions already locked (do not
relitigate): SQLite = system of record; DuckDB = optional read-side accelerator,
turned on ~10–15k rows (measured); Parquet = export/cold tier; new shapes land in
JSON `payload`/`extra`, promote-when-hot.

---

## The load-bearing invariants (violate any → a dashboard breaks)

1. **ONE `run_id` string, byte-identical across store, Grafana label, and Langfuse
   `trace_user_id`.** Never fold user/host/benchmark scope into it. If a push
   namespace is ever needed, compose `{owner}__{run_id}` ONLY at the
   Pushgateway/Langfuse layer, identically on both sides, never in the stored id.
2. **`span_id` round-trips raw and byte-exact.** Two shapes coexist —
   `production_run::task` (has `::`) and `terminal-bench/task/turn_N_kind` (no `::`).
   `task_id` derivation must match the exporter's existing rule for BOTH.
3. **Measurement `payload` round-trips byte-equal.** `node_id`/`kind`/perfspect-TMA
   keys are NOT promoted columns — the exporter and dashboards read them from the
   JSON payload. NULL ≠ 0 (a missing key stays NULL, never 0).
4. **No store-only flip until the store is provably populated AND at parity.** The
   legacy JSON / `docs/*` / `/tmp` read paths stay as a verbatim fallback until then.
5. **Viewers open read-only; never run DDL/migrations.** Writers (runner, sweep)
   are the only ones allowed to migrate.

---

## Phase order (each phase: shippable, dashboards verified unchanged)

> **Single highest-risk ordering rule (the #1 verifier finding):** the
> store-switch for Grafana (`start.sh`/exporter) and EVERY store-only flip must
> occur **only after P10 backfill proves parity**. Until then, `docs/*` glob + JSON
> fallback remain the live source. P7 *adds* a store read path behind a flag; it
> does **not** remove the JSON loop.

### P0 — Schema versioning + connection hardening (no shape change)
- `PRAGMA user_version` + ordered `storage/migrations/000N_*.sql` runner; **fail
  loud** (raise, never swallow). Baseline current schema = v1.
- `_get_connection()`: `PRAGMA foreign_keys=ON` + `journal_mode=WAL` on **every**
  connection (FK pragma is per-connection). Add `read_only` → opens `file:…?mode=ro`,
  skips `_ensure_db`/DDL.
- **FIX (live #10):** make `live_dashboard`'s sweep-DB opens `read_only=True` **now**,
  not in P6 — today it opens `SQLiteResultStore(_cand)` as a writer at module top
  level and would run migrations + hold the WAL write lock against a live sweep.
- `.gitignore` `*-wal *-shm`.
- *Verify:* `user_version>=1`, `fk=1`, `journal=wal`; legacy `docs/*.db` opens
  read-only without mutating (git status clean); `trace/test_builder.py` passes.

### P1 — Real `store_measurements` + `measurements` table (DUAL-WRITE)
- Replace the no-op: one upsert/record into `measurements` (6 promoted NULLable hot
  cols + full `payload` JSON + `seq` for insertion order). `query_measurements`
  returns rows reconstructible to `MeasurementRecord` with **payload byte-equal**.
- Runner still writes `measurement_records.json`; all readers still read JSON.
  Nothing flips.
- **FIX (grafana #13):** assert the 4 `perfspect` records have **distinct
  `span_id`** before relying on `UNIQUE(run_id,span_id,layer)` — else the upsert
  silently collapses them and the TMA row blanks. Widen the key if not distinct.
- *Verify:* round-trip test on **real `docs/*` L3 + perfspect payloads** (not just
  synthetic); `count(*)>0`; NULL-vs-0 preserved; JSON still written; Grafana unchanged.

### P2 — Thread ONE `run_id` end-to-end (pure wiring, no shape change)
- `run_tb2_with_storage.py` etc.: pass store `run_id` into `RunContext(run_id=…)`;
  kill divergent uuid paths; source `hardware_sku` from `platform.detect` (drop
  `'8592+'` literal).
- `harbor_sweep.py`: insert a real `runs` row for each per-cell run_id **before**
  `store_sweep_point`.
- **FIX (live #3, CRITICAL):** also insert a `runs` row for the **`sweep_id`
  itself** — the scaling verdict is stored under `run_id=sweep_id` with no `runs`
  row, so P3's cascade FK + fail-loud rebuild would reject it and crash the whole
  live page. (Alternative: store the scaling verdict under a real run_id.)
- **FIX (langfuse #7):** isolate the `litellm_terminal_loop.py` span-CM change from
  the adjacent `completion_metadata()` call; verify `parent_span_id`(=task_id=
  Langfuse trace_name/session_id) is unchanged, not just `run_id`.
- *Verify:* `run_id` identical across measurements/task_results/spans; no dangling
  `sweep_points.run_id` or `analyzer_verdicts.run_id`; all 4 dashboards on old paths.

### P2.5 — Repoint the sweep WRITE path (NEW — fills verifier gap #2)
- **FIX (live #2, CRITICAL):** move `harbor_sweep.py` / `run_scaling_sweep_tb2.py`
  off `SQLiteResultStore(spec.output_dir)` (=`/tmp/agentsysperf_sweep`) onto
  `ResultStore.open()`/`persist_run`. Without this, P6 repoints the live scaling
  *read* path to `$AGENTSYSPERF_HOME` while sweeps still *write* `/tmp` — the live
  dashboard's whole purpose (on-demand sweeps) shows nothing.
- *Verify:* a freshly-run sweep appears in `$AGENTSYSPERF_HOME` and renders in the
  live scaling pane via the DAL.

### P3 — Structural migrations (FK cascade, composite task PK, provenance, benchmarks)
- `0003`: rebuild `task_results` with composite PK `(run_id, task_id)` +
  `logical_task_key`; add `ON DELETE CASCADE` on `task_results/spans/
  analyzer_verdicts.run_id` and `sweep_points.sweep_id`.
- `0004`: `benchmarks` table; `runs.benchmark_id/optimization_profile/numa_policy/
  agentsysperf_version/result_digest/owner_id/owner_kind/host_id`; **drop the
  `hardware_sku` literal default** (source from `detect_platform()` at write).
- `0006`: `analyzer_verdicts` UNIQUE `(run_id,task_id,analyzer_name)` upsert.
- **FIX (live #5, CRITICAL):** `0006`'s table rebuild must **clean/tolerate
  dangling sweep-verdict rows** (run_id=sweep_id) BEFORE adding the cascade FK,
  or the fail-loud migration crashes every store open including read paths. (P2's
  sweep_id `runs` row is the primary fix; this is the belt-and-suspenders.)
- Tolerate legacy DBs with **no `spans` table** (`docs/agentsysperf_results.db`).
- *Verify:* migrate a **copy** of `docs/*.db`; composite PK lets same task_id
  survive under two run_ids; `delete_run` cascades with no orphans; raw-sqlite3
  report scripts still parse (pin explicit column lists); committed `docs/*.db`
  **not** mutated by tests.

### P4 — `ResultStore.open()` seam + widen the Protocol
- `@classmethod open(*, dsn=None, read_only=False)`: resolve `dsn` arg > 
  `$AGENTSYSPERF_STORE_URL` > `$AGENTSYSPERF_HOME/results.db` (default `~/.agentsysperf`).
  Zero-arg `open()` = local SQLite; a Postgres DSN = future plugin (raise clear
  `NotImplementedError` for now — seam exists, backend later).
- Fix `_discover()` to dispatch by **URL scheme** calling `cls.from_dsn(dsn)`, not
  the broken zero-arg `cls()`.
- **Widen the `ResultStore` Protocol** to the full real surface (store/query_spans,
  6 sweep methods, list_benchmarks/list_runs/get_run/latest_run/query_measurements/
  delete_*/persist_run/store_artifact/get_artifact_path) **before** any server work,
  so a Postgres backend implementing only the Protocol loses nothing.
- Keep `__init__(output_dir)` for back-compat.
- **FIX (#8, HIGH):** pin ONE canonical `$AGENTSYSPERF_HOME` used by backfill AND all
  three long-running services. Do **not** use `/tmp` for the persistent store
  (verify commands that used `/tmp/ap_home` are test-only).

### P5 — Runner atomic write-on-stop via `persist_run` (DUAL-WRITE kept)
- `RunContext.stop()`: one transaction → run+tasks+measurements+spans+verdicts.
  Keep writing JSON.
- **FIX (live #7, MEDIUM):** `live_dashboard` must **explicitly** set
  `AGENTSYSPERF_NO_STORE` (or pass `store=None`) — the "default None → open() lazily"
  behavior would otherwise persist per-click `run_<ts>` churn into the canonical
  store and add write-lock contention on every click.
- *Verify:* atomic (forced mid-flush exception → zero rows); synthetic run readable
  via new methods; JSON still written; dashboards unchanged.

### P6 — `dashboard_data` DAL + repoint demo & live (DB-first, **verbatim** JSON fallback)
- New `src/dashboard_data.py`: only data source; opens `read_only` per call;
  caches **plain dicts/lists, never a connection**. Each method tries store, else
  lifts the **exact** legacy `/tmp`+`docs/*` probe verbatim.
- **FIX (demo #4, CRITICAL):** reconcile the hardcoded `_NAV_PAGES`
  `"{Section}::{Page}"` dispatch with store slugs — keep a **slug↔display-label
  map** so dispatch keys stay literal. Data-drive the *data*, not the labels, or the
  whole app goes blank.
- **FIX (demo/live #11):** `latest_run` must reproduce the dashboards' **curated
  first-match priority** (prefer real EMON over dry/benchmark), not "newest run_id".
  Add a per-benchmark *preferred/pinned run* selection + an explicit **latest-sweep**
  method (sweeps order by `created_at`, a different key than runs).
- **FIX (demo #2/#7, HIGH):** for file-addressed artifacts (EMON CSV, scaling PNG)
  the DAL fallback must lift `live_dashboard`'s **specific** dir+pattern set (it
  differs from demo's). EMON page keeps a **real** `csv_path`; `EmonAnalyzer` needs
  an on-disk file.
- **FIX (demo #8 / langfuse #16, HIGH):** **leave the Langfuse page on `os.environ`** —
  do NOT route its host/keys through the DAL. It has zero store coupling; a DAL
  returning None silently disables it, and its unguarded config block would crash
  the page on the P10 flip. Keep the REST fetch **uncached**.
- *Verify:* parity harness asserts DAL rows/findings == old-loader output per
  benchmark (no zeros where data existed); every page renders incl. EMON/Scaling/
  Recommendations; **with empty `$AGENTSYSPERF_HOME` both apps still render from
  legacy fallback**; Recommendations finding-**count** parity (its bare
  `except:pass` masks regressions — also narrow those excepts).

### P7 — Grafana exporter gains a store read path (BEHIND A FLAG; JSON loop stays)
- `export_run_from_store(store, run_id)` rebuilds the SAME exposition text as
  `export_from_file` — metric names, label keys, `span_id`→`task_id` rule,
  `node_id`→workload, `kind`, `sku` **verbatim**. Reads `node_id`/`kind`/perfspect
  from the **payload**, not promoted columns.
- **FIX (#1, CRITICAL — the dominant ordering bug):** do **not** replace the
  `docs/*` loop in `start.sh` here. Add the store path behind a flag; the `docs/*`
  glob remains the live source until **after P10** proves parity. (P7's prose and
  its own risk note contradicted; the fallback wins.)
- **FIX (grafana #1, CRITICAL):** add a verify that runs `export_run_from_store`
  against a **backfilled `docs/tb2_phase2_test`** and confirms the **phase row
  populates under the SAME run_id** as the hardware row — today verdict/span/
  hardware align only by naming coincidence.
- **FIX (grafana #14):** run the exposition-diff parity against **both** span_id
  shapes (`::` and `/`) **and** the perfspect/TMA dataset, not one run.
- **FIX (grafana #5):** pin the dashboard-facing `sku` label — decide literal vs
  detected vs `unknown (backfill)`; the diff will catch the change but the plan must
  choose. (Recommend: emit detected SKU; accept it's a new series; document it.)
- Keep the post-restart re-push (TSDB is ephemeral); add a `promdata` named volume
  later to reduce churn.
- *Verify:* `git diff --exit-code` on `agentsysperf_dashboard.json` (regen byte-stable);
  exposition identical old-vs-new across both span shapes + perfspect; run_id
  dropdown values unchanged.

### P8 — DuckDB optional analytics extra + Parquet export (added, never blocking)
- `[tool.poetry.extras] analytics = ["duckdb","pyarrow"]`, **pinned** duckdb (sqlite
  scanner skew). Guarded import; `agentsysperf analytics`/`export --format parquet`
  give a clean "install agentsysperf[analytics]" message if absent.
- DuckDB `ATTACH`es the one `results.db` (zero copy); Parquet for cold-tier/publish.
- *Verify:* attached count == SQLite count; parquet round-trips; **without** the
  extra, core + all 4 dashboards unaffected; P5 persist_run test still green.

### P9 — Script-sprawl collapse to CLI
- `agentsysperf run|report|db ls/show/rm|serve|migrate|tasks`. Fold the duplicate
  `run_tb2_*`/`run_emon_*`/report generators; **delete** byte-dupes; **move** pptx
  deck builders to `scripts/decks/`; `report` drives `XeonPowerPointGenerator` via
  the ResultStore API (never raw sqlite3).
- **FIX (langfuse #15, MEDIUM):** add a verify that `AGENTSYSPERF_LANGFUSE`/`LANGFUSE_*`
  env **propagates** through `agentsysperf run` (if it execs a subprocess/sanitized env,
  tracing silently dies with only a warning).
- Don't delete `generate_simple_demo_report.py`'s source DB until its report is
  regenerated via `agentsysperf report`.

### P10 — Idempotent backfill, parity gate, THEN flips
- `scripts/migrate_to_unified_store.py`: idempotent, **fail-loud**, takes
  **caller-provided `--source-root` (repeatable) / glob** — NOT a hardcoded `docs/*`
  (colleague GNR runs live at unknown paths). **Dry-run by default:** scan roots,
  report every DB + JSON dir with table/row counts + detected format, migrate
  nothing until `--apply`. Reads every source **`read_only=True`** and copies into a
  **fresh** `$AGENTSYSPERF_HOME` store — **originals byte-identical** (md5 pre/post
  asserted), never upgraded in place. UNION scattered DBs (dedup composite keys),
  seed `benchmarks` from a **slug map** (must include `swe_bench`/`tau_bench` and
  edge dirs — **FIX #12**), backfill measurements from JSON, register EMON CSV/
  scaling PNG/`all_results.json`/`analysis_report.json` as artifacts.
- **FIX (demo #1, CRITICAL):** persist the **synthetic** dataset into the store
  (`examples/run_synthetic_l1_l3.py` through `RunContext(store=open())`) and treat it
  as the canonical Terminal-Bench Results & Metrics + Bottleneck source — backfill
  only covers `docs/*`, so without this both flagship pages go empty on flip.
- **FIX (#9, HIGH):** before any store-only flip, assert `get_artifact_path` resolves
  to an **existing file** for EMON/Scaling pages — count-equality (0==0) can't tell
  "legitimately empty" from "fallback removed, file gone".
- **FIX (#1):** ONLY NOW flip `start.sh`/exporter to store-driven and remove DAL
  fallbacks — gated on `verify_backfill_parity.py` passing.
- **FIX (demo #12):** Tau/SWE raw-results panes — either define a `results-json`
  artifact kind, or **explicitly leave those panes on the legacy reader** (don't flip
  globally).
- **Retire** `tb2_db_to_records.py` only after backfill verified and Grafana/DAL read
  from the store. Never delete `docs/*` JSON or DBs (cold backup).
- *Verify:* re-run migrate → zero new rows (idempotent); parity counts ≥ sources;
  full-stack render identical from the single store; HARD RULES — assert
  `harness/results/`, `wss_orchestration_example/` untouched.

### Colleague / GNR existing-runs migration (decided 2026-06-11)

Three data populations, three answers:

1. **`harness/results/` (GNR baselines, 18+ datasets):** Harbor-format JSON/CSV/txt,
   NOT agentsysperf SQLite/MeasurementRecord data. **Never read or written** by any
   phase (HARD RULE + format mismatch). Keeps working unchanged — cannot break
   because the storage code never opens it.
2. **Existing `agentsysperf_results.db` files (incl. colleague's GNR DBs):** verified
   on copies of all 9 local DBs — P0–P1 auto-migration upgrades them to
   `user_version=2` + adds `measurements` with **zero row loss, no FK failures**.
   Sweep DBs are the exception (dangling `run_id=sweep_id` verdict rows) — fixed by
   P2 (insert sweep_id `runs` row) + 0006 cleanup BEFORE the P3 cascade FK lands.
3. **`measurement_records.json`-only runs:** measurements were never in any DB;
   backfill ingests them net-new.

**DECISIONS (user, 2026-06-11):**
- **NEVER mutate originals.** `scripts/migrate_to_unified_store.py` opens every
  source `read_only=True`, copies into a **fresh** canonical store, leaves each
  original DB **byte-identical** (verified via md5 pre/post). Originals stay as a
  fallback. No in-place upgrade of colleague data.
- **Source roots are CALLER-PROVIDED, not assumed `docs/*`.** The colleague's GNR
  runs live at unknown paths (their `$AGENTSYSPERF_HOME`, `/tmp`, or `--output` dirs).
  The migrate CLI takes `--source-root <path> [...]` (repeatable) / a glob, and
  defaults to a **dry-run discovery mode**: scan the roots, report every DB + JSON
  dir found with table/row counts and detected format, and migrate nothing until
  `--apply`.
- **CAVEAT — P0 auto-migrate-on-open is in-place.** Opening a DB with the new code
  as a *writer* upgrades it (one-way: `user_version` bump). For colleague data this
  must be avoided: the migrate path opens sources `read_only` (no upgrade), and we
  must NOT point a writer (runner/dashboard) at the colleague's original DB. Document
  loudly; the canonical store is the only writable target.

### Rollback (verifier gap — was missing)
Each flip removes the only working fallback, so: **back up `$AGENTSYSPERF_HOME/results.db`
before every migration**; document the revert path for each flip (DAL→legacy
fallback; `start.sh`→`docs/*` glob; restore DB copy). Schema rebuilds (0003/0006) are
the highest-risk DDL — copy-first, transaction, fail-loud.

---

## Database-as-a-service / multi-user (LATER tier, same seam)

**Defer the server; build only the seam now.** Tier 0 (local zero-config SQLite WAL
file) is the only tier the single researcher needs. Introduce a server **only** when
a concrete trigger fires: (a) >1 host writes the same store, (b) sustained concurrent
writers exceed WAL's single-writer comfort, or (c) per-user isolation/audit is
needed. Bare-metal/managed **Postgres reachable by DSN** delivers "one endpoint many
hosts point at" with least ceremony; also ship a docker-compose `store` profile.

- **Seam:** the same `ResultStore.open(dsn=)` call site; flip is
  `export AGENTSYSPERF_STORE_URL=postgresql://…` with **zero caller changes**. Postgres
  backend ships as an out-of-core entry-point plugin (scheme `postgres`).
- **Multi-user:** owner identity is a **column** (`owner_id/owner_kind/host_id`),
  auto-populated from `$USER`/hostname locally — **never folded into `run_id`** (that
  would desync Langfuse + Pushgateway). ULID run_ids let multiple hosts write
  concurrently without collision. `list_runs(owner_id=None)` = the shared
  leaderboard; `owner='me'` = my experiments. Deletes owner-scoped at Tier 1+ via
  Postgres roles + token-in-DSN (no bespoke auth server). Published/withdrawn uses
  the `status`/`result_digest` tombstone.
- **What goes where:** relational system-of-record → SQLite→Postgres; bulk artifacts
  (EMON CSV, PNGs) → `runs/<run_id>/` files → later an `ArtifactStore` (local|s3)
  mini-protocol; Parquet → publish/cold tier; **Langfuse keeps its own
  ClickHouse/S3 — never merged**, the agentsysperf store holds only the `run_id` join
  key (+ optional langfuse host/project ids in `runs.metadata` for deep-linking).
- **docker-compose profiles:** `store` (Postgres+pgdata, off by default),
  `dashboard` (demo+live), `monitoring` (prometheus+pushgateway+grafana+`promdata`
  volume + a store-driven pusher), `langfuse` (web+postgres+clickhouse+redis+minio,
  separate datastores).

---

## Verifier scorecard (first draft → corrected)

| Dashboard | First-draft verdict | After fixes |
|---|---|---|
| demo :7860 | SAFE-WITH-GAPS | gaps #1,2,4,7,8,11,12,16 folded into P6/P10 |
| live :7861 | **UNSAFE** | CRITICALs #2,3,5,10 → P0/P2/P2.5/P3 |
| grafana :3000 | SAFE-WITH-GAPS | CRITICAL #1 ordering + #5,13,14 → P7/P10 |
| langfuse :7862 | SAFE-WITH-GAPS | HIGH #6,#16 → keep run_id canonical, env on os.environ |

**DuckDB/Parquet (P8) correctly optional & non-blocking. No HARD-RULE violation.**
