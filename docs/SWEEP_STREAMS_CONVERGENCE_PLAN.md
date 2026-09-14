# Converging the density sweep onto the streams substrate

**Status:** ⛔ **BLOCKED — do not start S1.** A 4-agent review found the two
substrates measure different experiments, and the plan below assumed they did
not. Resolve **S0** (new section) first. · **Date:** 2026-08-05 (revised)
**Prereqs:** P3 (artifact registry) and P4 (no silent zeros) — both landed.

## ⛔ Why this is blocked

The original plan treated the two runners as the same experiment with different
instrumentation. They are not. Two findings, each verified directly against the
code rather than taken from the review:

**1. The density axis means different things.** The bash runner writes ONE
Compose overlay, `cpuset: "$CPUS"`, applied to every agent in the cell
(`run_density_test_cwf.sh:150-158`) — N agents contend on one shared 16-core
cpuset. Streams `--pin-cores` carves DISJOINT cpusets, one per worker slot:

```
plan_task_slots(16, {1: 8, 2: 2})  ->  13 slots, cpusets 0-1, 2-3, 4-5, 6, 7, ...
                                       16 CPUs total, zero overlap
```

`_take_contiguous` removes each assigned CPU from the free list
(`resources.py:175-210`), so no two workers ever share a core. Same 16 cores,
same agent count, **opposite contention**: one is a scheduler-contention
experiment, the other is an isolation experiment. `ScalingAnalyzer` buckets by
density alone, so a dashboard holding both renders a knee describing no real
machine — the same pathology as `ScalingAnalyzer averages unlike silicon`.

**2. Per-worker perf cannot work on this box.** Streams spawns one
`mp_context.Process` per slot and starts them all
(`orchestrator.py:726-749`). The target platform has exactly 4 generic PMCs and the PMU has ONE
holder. 13 workers each opening a perf session means 13 sessions on 4 counters →
silent multiplexing, which is invisible in the CSV (a prior run measured
`counter_enabled_pct_min=64%` and read as if direct). Telemetry therefore cannot
be per-worker; it must be one session spanning the cell.

And streams has no cell boundary to span. `_worker()` pulls from a queue until it
receives `None` (`orchestrator.py:484-573`); there is no `.harbor_done` moment
like the bash runner brackets against (`run_density_test_cwf.sh:224-228`). A
"cell" would have to be synthesised as first-worker-start → last-result-collected,
which includes staggered spawn and the straggler tail.

Also confirmed: streams' admission gate checks affinity CPUs, Docker CPU/memory,
declared memory, proxy ports and network capacity — but has **zero** PMU checks
(`grep -c 'pgrep\|perf\|emon'` over `_admit_stream_capacity` → 0), while the bash
runner gates on `pgrep -x perf` and `pgrep -x emon` in 3 places. Two orphaned
perf processes once held counters at 47% enabled across 30 cells.

## S0 · Decide what we are measuring (must precede S1)

The user's constraint is streaming benefits **and** cpuset sizing **and** the
existing telemetry. All three are achievable only under option (a).

| option | density basis | telemetry | streams benefits | verdict |
| --- | --- | --- | --- | --- |
| **(a) one slot per cell**, `--pin-cores`, slot = whole cpuset | shared cpuset — **comparable to bash** | full: 1 perf session, cpuset-scoped | admission + pre-pull + scoped cleanup, but N=1 worker | **recommended** |
| (b) multi-worker, telemetry minus PMU | disjoint cpusets — not comparable | vmstat + cgroup only (these DO move safely) | full | separate curve, never mixed |
| (c) redesign the basis around disjoint cpusets | new `vcpu_basis_kind` | full per worker only if 1 perf session | full | measures a different experiment; needs its own reference curve |

**Recommend (a).** It keeps the axis comparable to `sweep_fib_clean`, keeps every
telemetry source valid, and still buys the four execution fixes that motivated
this. The cost is that a cell runs one worker, so streams' multi-slot scheduling
is unused *for sweep points* — which is fine, because the sweep's variable is
agent count inside a cpuset, not worker count.

Under (a), `--sweep-point-dir` must **require** `--pin-cores` and refuse when the
plan yields more than one slot. Under (b) or (c), the bash runner **cannot be
retired** — it stays the only path producing cpuset-pinned shared-contention
curves with full hardware telemetry, and S4 is struck.

## The problem in one line

We have two density-sweep pipelines. One can measure hardware and cannot execute
safely; the other executes safely and measures almost nothing. Neither is
complete, and only one reaches the dashboard.

## Measured today

| | `harness/scripts/run_density_test_cwf.sh` | `run-streams` + `sweep_total_cores.py` |
| --- | --- | --- |
| lines | 903 (+206 cpuset_telemetry, +301 container_telemetry) | 825 (orchestrator) + 169 (wrapper) |
| driver | `harbor run` per cell | `agentsysperf run-streams` |
| concurrency | `-n N --n-tasks N` per cell | worker slots, one queue per task CPU size |
| **core PMU** | `perf stat -C <cpuset>`, 4 events for the platform's 4 PMCs | none |
| **memory BW** | `perf stat -a uncore_imc/cas_count_{read,write}/` | none |
| **disk** | vmstat `bi`/`bo` | none |
| **network** | host counters | none |
| **cpuset CPU%** | `cpuset_telemetry.py` (`/proc/stat` over the pinned range) | none |
| **per-container cgroup** | `container_telemetry.py` (`cpu.stat`, `cpu.max`, `cpu.pressure`, `memory.peak`, `io.stat`) | none |
| latency percentiles | p50/p95 agent, verifier prologue vs pytest split | p50/p95/p99 from the DB |
| **image pre-pull** | no — builds land inside the timed window | **yes** (`prebuild.ensure_task_images`) |
| **admission gate** | no | **yes**, fail-closed: affinity CPUs, Docker CPU/mem, summed declared memory, proxy ports, Docker network capacity |
| **enforced per-task sizing** | parses `task.toml` cpus, hopes | **yes**, compose override → cgroup `cpu.max` / `mem_limit` |
| **scoped cleanup between cells** | no | **yes**, per-run containers + networks, no global prune |
| PMU pre-flight | `pgrep -x perf` gate (2 sites) | n/a |
| output | `point.json` per cell (74 keys) | `results.{json,csv}` (14 keys) |
| **reaches the dashboard** | **yes** — `sweep import` → `sweep_points` | **no** — 0 calls to `store_sweep_point` |

Two facts worth stating plainly:

- `grep -c streams run_density_test_cwf.sh` → **0**. The shell runner has no
  knowledge of the streams path.
- `grep -rn store_sweep_point src/streams/` → **0 matches**. Everything
  `run-streams` measures is invisible to the Scaling views, permanently, until
  someone writes that bridge.

## Why not just add telemetry to the wrapper

`sweep_total_cores.py` shells out to `agentsysperf run-streams` per point and
reads durations back out of SQLite. Wrapping *that* subprocess in `perf stat`
would measure the orchestrator's whole lifetime — including image pre-pull,
admission probes, and cleanup — not the measured task window. The shell runner
gets this right by bracketing only the `harbor run` call and terminating the
counters on a `.harbor_done` sentinel.

So the counters have to live where the task window is known. The original plan
said "inside `_worker()`" — **that is wrong**, and the review caught it: with N
workers that means N perf sessions on 4 PMCs. The session belongs to the CELL, at
the orchestrator level (`run_task_sized_streams`), opened before workers start and
closed after the last result is collected.

## Target design

Keep `point.json` as the contract. It is already the documented seam — the
importer's docstring says "any runner that emits `point.json` in the documented
shape can be imported, so this is not welded to one script" — and it already
carries all 74 fields the analyzers and dashboard read. The convergence is
therefore: **make streams emit `point.json`**, not: make the dashboard read a
second format.

Telemetry is opened ONCE per cell at the orchestrator level — not per worker.
Under S0(a) a cell is one slot holding N agents, so this is also the only place a
single perf session can cover the whole window.

```
run-streams --sweep-point-dir DIR --pin-cores        [refuses if plan != 1 slot]
  └─ per cell = one density point (N agents in ONE cpuset)
       ├─ PMU pre-flight gate      [NEW — pgrep -x perf / -x emon, port from bash]
       ├─ admission gate           [exists]
       ├─ image pre-pull           [exists]
       ├─ open telemetry ── ONE session per CELL  [NEW — lift from the shell runner]
       │    ├─ perf stat -C <cpuset>    4 core events (4 generic PMCs here)
       │    ├─ perf stat -a uncore_imc  memory BW (socket-scoped, label _socket)
       │    ├─ vmstat                   disk (drop first row)
       │    ├─ cpuset_telemetry.py      cpuset CPU% (--cpus <cpuset>, not host)
       │    └─ container_telemetry.py   per-container cgroup (--image-filter)
       ├─ run the task queue       [exists]        ← T0..T1 = the whole cell
       ├─ close telemetry          [NEW]
       ├─ assemble point.json      [NEW — port the parsers, all 74 keys]
       ├─ store_sweep_point + store_artifact ×8    [NEW]
       └─ scoped cleanup           [exists]
```

## Work breakdown

### S1 · Extract the telemetry session (no behaviour change) — ~1 day

Pull the open/close/parse logic out of the bash into a reusable Python
`TelemetrySession` context manager under `src/sweep/telemetry.py`:

```python
with TelemetrySession(cpuset="0-15", out_dir=cell) as tel:
    ...run the cell...
point.update(tel.rollup(elapsed_s=elapsed))
```

Must preserve, because each was a bug fix paid for once already:

- **4 perf events, not 10.** The platform has 4 generic PMCs; more forces silent
  multiplexing (a run measured 64% `counter_enabled_pct_min`).
- **Two perf sessions on purpose.** Core events on `-C <cpuset>`, uncore on
  `-a`. Different PMU, so they do not compete — both report 100% enabled.
- **`pgrep -x perf` gate, not `pgrep -f 'perf stat'`.** The `-f` form matched
  the runner's own launcher.
- **Scope labels are load-bearing.** `mem_bw_*_socket`, `disk_*_host`,
  `runqueue_is_host_wide` + `logical_cpus_host`. `uncore_imc` is socket-scoped
  and `-C` does **not** restrict it (measured). The importer's verdict basis
  depends on `runqueue_is_host_wide`.
- **Measured zero ≠ unmeasured.** `round(x, 1) if x is not None else None`,
  never `if x`. A 0 MiB idle-socket reading is a finding; absence is a gap.
- **Drop vmstat's first row.** It is an average since boot, not an interval
  sample, and swamps a short cell.
- **`--image-filter alexgshaw/` is mandatory** in `container_telemetry.py` on
  this shared box, or it samples other tenants' containers.

Two more the review surfaced, both silent-failure modes:

- **cgroup counters are cumulative.** `container_telemetry.py:204-209` takes
  `last - first`. Sampling once yields garbage, not a small number.
- **Refuse the cell, do not degrade it.** The bash runner rejects a cell when
  `len(trials) != n` or `completed != n` (`run_density_test_cwf.sh:441-446`): a
  hung trial inflated `elapsed_s` to 750s against a 50s norm and dropped
  throughput 10.4 → 1.8/min. Streams' wrapper never checks the returncode, so a
  failed cell would persist as a zero-throughput point on the curve.
- **`TelemetrySession` must gate on `pin_cores=True`.** Unpinned workers roam all
  288 CPUs, so a 16-core cpuset sampler reports ~5% while the box is pegged
  (measured elsewhere: 4.27% cpuset vs 0.43% host, ~10× understatement), which
  puts `CPU_SATURATION_PEAK=95` permanently out of reach and makes `cpu_bound`
  unreachable.

Verify — the original wording ("every one of the 74 keys must match within
noise") is **necessary but not sufficient**: keys can match while the basis
differs. The real gate is a same-experiment comparison:

> bash `N=16` on shared cpuset `0-15` **vs** streams one 16-core slot,
> `--pin-cores`, 16 agents in that slot. Accept only if the **knee density AND
> bottleneck class are identical**. A different knee means the substrate changed
> the measurement, which is the one outcome that invalidates the exercise.

Do not proceed past S1 without that comparison.

### S2 · `point.json` emitter in the worker — ~1 day

Add `--sweep-point-dir` to `run-streams`. Per **S0(a)** it must refuse unless
`--pin-cores` is set and the plan yields exactly one slot.

The **cell**, not the worker, writes `n{agents}_r{rep}/point.json` — a worker
cannot, because one perf session spans the cell and the counters belong to it.

Emission is not just a file. It must, mirroring `import_points.py`:

- `store.store_sweep_point(sweep_id=..., point=...)` — the dashboard reads
  `sweep_points`, and streams currently calls this **zero times**, which is why
  everything it measures is invisible.
- `store.store_artifact(...)` × 8 per cell (`_CELL_ARTIFACTS`), so a rollup stays
  auditable.
- Run `ScalingAnalyzer.analyze_sweep` at the end, with the runqueue basis chosen
  by `runqueue_is_host_wide` (host 288, not cpuset 16).

`_record_point_metadata` updates `runs.metadata` only, so it is **not** the seam
for this — the original plan was wrong to point at it.

Gate before the first real sweep: feed the emitted `point.json` to
`task_signature.signature_for_row()` and assert `.missing` is empty. Seven fields
are required — `quota_demand`, `quota_fill`, `serialization`,
`mem_bw_mib_s_socket`, `disk_write_mb_s_host`, `net_total_mb_s_host`, plus the
`cpu_avg` that `serialization` derives from. Missing any degrades a verdict to
`undetermined` **silently**, because `task_signature` omits rather than errors
while `ScalingAnalyzer._aggregate_replicates` zero-fills — and a fabricated
`cpu_avg=0.0` reads as `headroom_remaining` on a saturated box. Fail the port,
don't warn.

Fields streams can populate *better* than the shell runner, because it owns the
container lifecycle rather than inferring it:

- `ctr_quota_cpus` / `ctr_quota_fill_mean` — from the compose override it wrote,
  not parsed back out of `task.toml`
- `cell_status` — from `execution_failures` / `oracle_failures`, which is a
  real crash-vs-oracle distinction the shell runner cannot make
- `concurrency_reachable` — from admission, which already computed it

### S3 · Cross-validate against the trusted curve — ~1 day, mostly waiting

Re-run `circuit-fibsqrt`, 5 densities × 3 replicates, on a verified-quiet box
(the `quietwatch.csv` pattern from the v3 run: 565 samples, load1 median 0.02,
zero foreign containers). Compare against `sweep_fib_clean`, the trustworthy
reference: 15/15 cells, 3 reps, 100% counters enabled, CV < 1%, knee at density
1.0 → `cpu_bound`.

**The slot geometry must be stated, not left to the planner.** Per S0(a): ONE
16-core slot with N agents inside it, matching the bash runner's shared cpuset.
`--pin-cores --total-cores 16` alone is ambiguous — with a `{1: 8, 2: 2}`
histogram the planner returns **13 disjoint slots**, which is option (c), a
different experiment. Verify the plan before the run:

```
plan_task_slots(16, hist)  ->  must be exactly 1 slot, cpuset 0-15
```

Accept only if throughput per density is within replicate CV, the knee lands at
the same density, and the bottleneck class matches. A different knee means the
substrate changed the measurement, which is the one outcome that invalidates the
whole exercise.

Also assert `counter_enabled_pct_min == 100` on every cell. Anything less means
a second PMU consumer appeared and the IPC/cache figures are scaled estimates
presented as direct counts.

### S4 · Retire the shell runner — ~0.5 day

**Conditional on S0.** Retire only under option (a), and only when all four hold:

1. S3's knee density and bottleneck class match `sweep_fib_clean`.
2. Every cell reports `counter_enabled_pct_min == 100`.
3. `task_signature.signature_for_row()` reports no missing components.
4. The dashboard cannot mix `vcpu_basis_kind` values on one chart.

Under (b) or (c) this step is **struck**: the bash runner remains the only path
producing cpuset-pinned shared-contention curves with full hardware telemetry,
and it keeps that job.

Leave the script in place with a header pointing at the CLI, keep `sweep import`
(the generic seam, predates both runners), and update the internal
concurrency-scaling findings.

**Total: S0 ~0.5 day (decision + one planner check), then ~3.5 days** under
option (a). S1 is the risk; S2 is larger than "mechanical" — it must reimplement
all 74 fields, call `store_sweep_point`, register 8 artifacts, and run the
analyzer; S3 is wall-clock.

## What this buys

- Sweeps stop timing image builds (pre-pull moves outside the window).
- A sweep cannot silently oversubscribe — admission is fail-closed. The
  contaminated run that cost a day was exactly this failure mode.
- Per-task cgroup sizing is *enforced*, not parsed-and-hoped.
- Containers are cleaned between cells, so cell N+1 does not inherit cell N's
  leftovers.
- One pipeline to maintain instead of 903 lines of bash plus 825 of Python that
  do not know about each other.

## Risks

| risk | mitigation |
| --- | --- |
| **Disjoint vs shared cpuset measures a different experiment** | S0(a): one slot per cell. `--pin-cores` alone is NOT enough — assert the plan yields exactly 1 slot |
| **N concurrent perf sessions multiplex 4 PMCs silently** | One session per cell at orchestrator level, never per worker. Assert `counter_enabled_pct_min == 100` |
| **No cell boundary for counter bracketing** | Under S0(a) the cell IS the run, so T0/T1 are unambiguous. Under (b)/(c) this stays unsolved |
| **Streams admission has zero PMU checks** | Port the bash `pgrep -x perf` / `pgrep -x emon` gates into `_admit_stream_capacity` |
| Telemetry port changes the numbers | S1's key-by-key match **plus** the same-experiment knee comparison; keys can match while the basis differs |
| Unpinned mode makes cpuset/socket counters meaningless | `TelemetrySession` gates on `pin_cores=True`; a 16-core sampler reads ~5% while 288 cores are pegged |
| Missing field degrades a verdict instead of erroring | Assert `signature_for_row().missing` is empty; `ScalingAnalyzer` zero-fills, so a fabricated `cpu_avg=0.0` reads as `headroom_remaining` |
| Dashboard mixes bases | Never `SELECT` across differing `vcpu_basis_kind`; `ScalingAnalyzer` buckets by density alone |
| perf and EMON contend | Existing `pgrep -x perf` gate + `AGENTSYSPERF_DISABLE_EMON`; PMU has one holder |
| Streams' 3600s default timeout hides hangs | Keep the `cell_status` gate requiring `completed == n`; check the returncode, which the wrapper currently ignores |
| `max_slots` default 4096 exceeds the Docker address-pool ceiling (~28) | Enforce the ceiling in admission; the bash runner caps at 27 |

## Explicit non-goals

- No new dashboard format. `point.json` → `sweep_points` stays the only path.
- No merge of `run-streams` and `agentsysperf run`. Different purposes:
  throughput vs correctness. That distinction is the user's standing decision.
- No auto-populate command. Running a benchmark, a sweep, and EMON stay
  separate manual steps, because they measure different things for different
  reasons.

## Appendix — the two items that come first

**P3 · Kill the `/tmp` reads; use the artifact registry.** 22 of 26 hardcoded
`/tmp` paths in `demo_app.py` are dead, and `/tmp` clears on reboot. The
`artifacts` table is empty (`artifact kinds: NONE`), so `get_artifact_path`
always falls through to those dead paths. Register artifacts on write, read from
the store, keep `/tmp` as a last resort only.

**P4 · Ban silent zeros.** Replace `.get(key, 0)` with a helper returning `None`
and rendering "not measured". This defect class has now produced three
user-visible bugs: `throughput_trials_per_min` (a flat line at zero read as
"throughput does not scale"), `num_commands` (NULL for every run ever stored),
and the TMA panel (NaN drawn as four empty bars). Confirmed keys still read this
way against data that does not contain them:
`aggregate_throughput_turns_per_s`, `mean_throughput_turns_per_s`,
`total_duration_s`, `emon_metrics_count`, `logical_cpus`.
