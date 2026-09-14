# AgentSysPerf

A pluggable benchmark suite for end-to-end agentic AI stacks. Stack-level, not single-endpoint. Plugins for benchmarks, measurements, optimization profiles, and hardware telemetry compose through Python entry points so external developers can ship adapters without forking the suite.

## Status

Active runtime + working harness. Three parts, all live on `main`:

- **Runtime** (top-level `src/` Python package) — Protocols, plugin discovery, CLI, reference plugin implementations, analyzers, trace/StepTrace, storage, exporters, and Streamlit dashboards.
- **Harness** (`harness/`) — synthetic CPU workloads, perf-stat sampling with cgroup attach, NUMA / hugepages / isolcpus profile applier, replay proxy, and 18+ directories of measured baseline data on Xeon 6787P (Granite Rapids). Runs unchanged.
- **Methodology** (`docs/methodology/DESIGN.md`) — the 16-section design rationale, plus the integrity principles every plugin honors (stated under Methodology references below).

8 Protocols govern plugins: BenchmarkAdapter, AgentInvoker, Measurement, OptimizationProfilePlugin, HardwareTelemetryPlugin, Analyzer, ResultStore, ReportGenerator. What's wired today:

- **6 BenchmarkAdapters** — synthetic_cpu (9 workloads), terminal_bench (Harbor/Terminus-2), openclaw, swe_bench, tau_bench, vllm_inference.
- **4 Measurements** — l1_subspan (per-span psutil), l1_system (node-level CPU/runqueue/ctx-switch/mem from /proc), l3_perf (perf-stat, target = system / self / pid:N / cgroup:PATH), perfspect (TMA, Intel). Deeper Intel TMA/EDP telemetry (EMON) is available on Intel hosts through a separate plugin and is not part of the vendor-neutral core.
- **7 Analyzers** — cpu_bound, cache, memory_leak, memory_bandwidth, breakdown, phase_profiler, scaling (concurrency knee + bottleneck).
- **6 OptimizationProfilePlugins** — base + 5 Xeon variants; declarative specs only today, `agentsysperf run` does not yet apply or engagement-verify a profile (see the status note under Examples). **1 HardwareTelemetryPlugin** — perf_stat (generic events; vendor-aware telemetry is a future plugin).
- **Storage + reporting** — a single canonical SQLite result store at `$AGENTSYSPERF_HOME/results.db` (default `~/.agentsysperf`; runs / task_results / measurements / spans / analyzer_verdicts / sweeps / benchmarks / artifacts), schema-versioned with auto-migrations; PowerPoint **and** Markdown report generators, Prometheus/Grafana exporters, Langfuse step-trace export, and an optional DuckDB analytics accelerator + Parquet export.
- **CLI** — `agentsysperf run` (benchmarks, with record/replay LLM caching), `run-streams` (task-sized Terminal-Bench streams), `report` (md/pptx), `db ls|show|rm`, `analyze`, `analytics`, `export`, plus per-group `list` drill-downs.
- **Dashboards** — `demo_app.py` (benchmark pages incl. Concurrency Sweep + Density Study) and `live_dashboard.py` (runs real workloads). Both read the canonical store via a shared data-access layer, with a verbatim fallback to legacy JSON during the transition.

## Repository Structure

```
src/                  Python runtime package
  protocols.py              8 Protocols: BenchmarkAdapter, AgentInvoker, Measurement,
                            OptimizationProfilePlugin, HardwareTelemetryPlugin,
                            Analyzer, ResultStore, ReportGenerator
  cli.py                    run / run-streams / report / db (ls|show|rm) /
                            analyze / analytics / export / list + per-group drill-downs
  home.py                   $AGENTSYSPERF_HOME resolution (canonical store + artifacts)
  benchmarks/               6 adapters (synthetic_cpu, terminal_bench, openclaw,
                            swe_bench, tau_bench, vllm_inference)
  measurements/             l1_subspan, l1_system, l3_perf, perfspect
  analyzers/                cpu_bound, cache, memory_leak, memory_bandwidth,
                            breakdown, phase_profiler, scaling
  optimization_profiles/    6 reference profiles + shared scaffolding
  replay/                   record/replay proxy + ReplayProxy + fixture validation
  sweep/                    concurrency-scaling sweep (SweepSpec, HarborSweep)
  storage/                  SQLite result store + schema + migrations/ +
                            optional DuckDB analytics (incl. sweeps/sweep_points,
                            measurements, benchmarks, artifacts)
  trace/                    StepTrace schema + Langfuse export
  exporters/                Prometheus / Grafana
  dashboard/                Plotly figure builders for the Streamlit views
  reporting/                PowerPoint report generator
demo_app.py, live_dashboard.py   Streamlit dashboards (ports 7860 / 7861)
docs/methodology/           DESIGN.md, hw_metrics_catalog.md, example_output.md
docs/contracts/             Original 12-ABC reference, 11 routing strategies,
                            study YAMLs (kept as historical reference)
examples/                   End-to-end driver scripts
experiments/scaling/        Density-contention experiment (synthetic workers)
harness/                    Working benchmark harness; results/ holds
                            measured baseline data on Xeon 6787P
fixtures/                      Canonical replay fixtures (TB2 Sonnet, etc.)
docs/benchmark/                Replay methodology documentation
```

## Quick Start

### Prerequisites

- **Python 3.12 or 3.13.** The floor is hard: `harbor` and `numpy` both publish
  no wheels below 3.12, and the resolver reports that as
  `No matching distribution found for numpy>=2.4.0` — a message that never
  mentions Python. `pyproject.toml` allows `>=3.12,<3.15`; 3.12 and 3.13 are
  the versions this suite is tested on. Stock Ubuntu 22.04 gives you 3.10, so
  install 3.12 explicitly (`apt install python3.12-venv`).
- **Linux on x86_64.** The portable measurements (`l1_subspan`, `l1_system`)
  read `/proc` and need no privileges; `l3_perf` shells out to Linux `perf` and
  needs `kernel.perf_event_paranoid <= 1`; platform detection uses `lscpu`. The
  `perfspect` measurement additionally requires Intel tools (PerfSpect) and is
  opt-in — a run without it still completes, with fewer measurement layers.
  Deeper Intel EMON/EDP telemetry is a separate plugin, not core.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .

# What's installed?
agentsysperf list

# Per-group drill-downs
agentsysperf benchmarks list
agentsysperf measurements list
agentsysperf profiles list
agentsysperf analyzers list
agentsysperf telemetry list   # + which profile counters nothing can verify here

# Is this host ready for accurate measurements?
agentsysperf preflight
```

`preflight` checks perf_event_paranoid, CPU governor, NMI watchdog, NUMA tools,
PerfSpect availability, and kernel headers. Fix any issues it flags before
running benchmarks — results collected under `powersave` governor or with
`perf_event_paranoid > 1` will be incomplete or misleading.

Run a benchmark and inspect it — results land in the canonical store
(`$AGENTSYSPERF_HOME/results.db`, default `~/.agentsysperf`) and are immediately
visible to the dashboards and `report`:

```bash
# Synthetic CPU workloads — no LLM, no network, no API key
agentsysperf run --benchmark synthetic_cpu --num-tasks 3

# Terminal-Bench — requires Docker/Harbor + OPENAI_API_KEY
export OPENAI_API_KEY=sk-...   # your OpenAI API key
agentsysperf run -b terminal-bench --num-tasks 2 --model gpt-4o-mini

agentsysperf db ls                      # runs in the store, newest first
agentsysperf db show <run_id>           # metadata + tasks + measurement layers
agentsysperf report <run_id> --format md   # or --format pptx
agentsysperf db rm <run_id>             # delete a run (cascades its data)
```

**Every run reports its own coverage.** Which measurements produced data and
which analyzers spoke depends on the host — `l3_perf` needs `perf` access,
`perfspect` needs Intel tools — so a run states what it got and why it
missed the rest rather than leaving a partial run looking complete:

```
Run synthetic_cpu_1785382672: 3/3 passed, 7 records.
measurements 2/4 active (l3_perf: no records for any span; perfspect: no records for any span)
analyzers 3/7 emitted (cache: needs layer(s) l3; cpu_bound: needs layer(s) l3; ...)
```

Counts are derived from what actually reached the store, not from what was
discovered, so a probe that registered and collected nothing reads as silent.
Add `-v`/`--verbose` for the INFO-level detail behind each skip.

**LLM replay caching** — record an LLM trajectory once, replay it free
thereafter (skips the cost/latency/variance, still measures hardware):

```bash
agentsysperf run -b terminal-bench --tasks task-a,task-b --record /tmp/tb2.jsonl   # live (records turns)
agentsysperf run -b terminal-bench --tasks task-a,task-b --replay /tmp/tb2.jsonl   # free, deterministic
```

Note: `--replay` requires `--tasks` (explicit task names that match what was recorded).
Use `-n` only with `--record` for first-time recording from the dataset.

### Quickstart: Populate all demo app tabs (offline, no API key)

Start fresh and fill the entire Terminal-Bench dashboard locally with one command sequence. All steps work offline — the `--dry-run` and `--record` workflows use cached data:

```bash
# 1. Synthetic CPU baseline (Hardware Baselines tab)
agentsysperf run --benchmark synthetic_cpu --num-tasks 9

# 2. Concurrency sweep (Concurrency Sweep tab, no API key needed)
#    --dry-run cells are SYNTHETIC modeled points, not measurements. They are
#    stored with data_source=synthetic and the dashboard badges them; drop
#    --dry-run (and add --fixture) for a real, measured sweep.
agentsysperf sweep run --dry-run

# 3. Launch dashboard — all tabs now populated
streamlit run demo_app.py --server.port 7860 --server.address 0.0.0.0
```

That's it. You now have a measured Hardware Baseline, a synthetic Scaling sweep, and system state visible. To populate the Agent Performance tab with **real** terminal-bench data, you need an LLM key:

```bash
# Optional: real agent runs (requires OPENAI_API_KEY + Docker/Harbor)
export OPENAI_API_KEY=sk-...
agentsysperf run -b terminal-bench --num-tasks 2 --model gpt-4o-mini
```

### Full Terminal-Bench workflow: all five views

The dashboard reads from four different data sources, so each view has its own population path:

| View | Data Source | Population Command |
|------|-------------|--------------------|
| **Hardware Baselines** | Synthetic CPU + L3 perf | `agentsysperf run --benchmark synthetic_cpu --num-tasks 9` |
| **Agent Performance** | Terminal-Bench + L1/L3 | `agentsysperf run -b terminal-bench --num-tasks 2 --model gpt-4o-mini` (needs OPENAI_API_KEY) |
| **Scaling (Concurrency Sweep)** | Sweep experiment | `agentsysperf sweep run --dry-run` (synthetic cells) or `agentsysperf sweep run --fixture <f.jsonl>` (real, measured, hours) |
| **Hardware Analysis** | Intel TMA telemetry | Requires the separate Intel-only EMON plugin (not included) |
| **Recommendations** | All above + analyzer verdicts | Auto-populated by all above; visible once any analyzer emits verdicts |

**Important caveats:**

- **A normal run collects all four core measurement layers.** If the optional
  Intel-only EMON plugin is installed, EMON needs exclusive PMU access — while
  the SEP driver holds it, `perf`-based layers report physically impossible
  counts (which the probes reject rather than average in), so EMON runs on its
  own pass, separate from the `perf`-based layers.
- **Terminal-Bench reports 4 pipeline phases, not 5.** `admit` (container
  provisioning + setup), `reason`, `act`, `commit`. There is no `retrieve` span
  because these agents do filesystem inspection (`grep`/`find`/`cat`) as shell
  actions — that is `act`. `retrieve` means semantic retrieval (embeddings,
  vector index); see `benchmarks/six_phase_agent` for a workload that has it.
  Phases with no spans are omitted, so the four shown sum to 100%.
- **`--replay` is not currently a zero-cost path for Terminal-Bench.** The
  committed fixture was recorded with a different agent, and `trial_key` hashes
  the first user message, so every lookup misses and the agent takes no actions.
  Record your own fixture first (`--record`), then replay it.
**Task-sized Terminal-Bench streams** — run the retained replay fixture with
the default 23-task selection and explicit unpinned worker slots:

```bash
agentsysperf run-streams \
  --slots 48 \
  --replay /path/to/your/fixture.jsonl \
  --ref 1
```

See the [full task-sized replay workflow](examples/terminal_bench_task_sized_replay/README.md)
for prerequisites, pinned scheduling, admission checks, cleanup, sweeps, and
timing semantics.

End-to-end examples (require Linux `perf` and `kernel.perf_event_paranoid=-1`):

```bash
# 9 synthetic CPU tasks with L3 perf-stat per span (per-PID attribution by default)
python examples/run_synthetic_with_l3.py --duration 2

# Optimization profile engagement-verify against the installed telemetry
python examples/run_profile_verify.py
```

**Status — optimization profiles are not yet wired into runs.** `agentsysperf run` has no `--profile` flag: it never calls `apply()` or `verify_engaged()`, and the `optimization_profile` field on a stored run is always empty. What ships today is the six declarative profile specs plus the engagement-verify machinery, exercised only by `examples/run_profile_verify.py`. Measuring one workload under `base` vs `amx_*` and attributing the delta — the reason profiles exist — is future work.

Running that example today, all 5 Xeon profiles fail engagement-verify with explicit "counter not advertised" failures. That is honest reporting rather than a bug, but the gap is a *naming* one as much as a hardware one: profiles declare derived metric names (`amx_active_cycle_ratio`, `hugepage_fault_ratio`) which are matched literally against a plugin's `available_events`, and 4 of the 6 declared counters are not PMU quantities at all — they come from `/proc/vmstat`, QAT sysfs, and oneDNN/OpenVINO instrumentation. Shipping vendor PMU telemetry (`intel_pcm`) is therefore necessary but not sufficient. See `docs/measurement_layers.md` and the naming contract on `HardwareTelemetryPlugin` in `src/protocols.py`.

## Plugins and discovery

Plugins are pure Python classes registered via entry points in their package's `pyproject.toml`:

```toml
[project.entry-points."agentsysperf.measurements"]
my_probe = "my_pkg.probe:MyMeasurement"
```

Seven entry-point groups are defined:

- `agentsysperf.benchmarks` — implementations of `BenchmarkAdapter`
- `agentsysperf.measurements` — implementations of `Measurement`
- `agentsysperf.optimization_profiles` — implementations of `OptimizationProfilePlugin`
- `agentsysperf.hardware_telemetry` — implementations of `HardwareTelemetryPlugin`
- `agentsysperf.analyzers` — implementations of `Analyzer`
- `agentsysperf.result_stores` — implementations of `ResultStore`
- `agentsysperf.report_generators` — implementations of `ReportGenerator`

`agentsysperf list` enumerates whatever pip-installed plugins advertise themselves on those groups. A plugin that errors on import is logged and skipped, not raised — a third-party plugin breaking does not break the CLI.

## Methodology references

- `docs/methodology/DESIGN.md` — full design document, 16 sections
- `docs/contracts/plugin_contracts.py` — original 12-ABC reference for context
- `docs/contracts/routing_strategies.py` — 11 reference routing strategies (paper)
- `docs/contracts/optimization_profiles.py` — original 6 profile reference (now ported to runtime)

### Integrity principles

Four rules every plugin honors. They are the whole reason to trust a number this
suite prints, so they are stated here rather than left to a design appendix:

- **Measured, not assumed.** A reported quantity comes from a counter, a clock, or
  a probe. Nothing is inferred from a nameplate spec or a per-generation constant —
  if the value cannot be measured on this host, it is reported as unknown, not
  estimated into a plausible-looking number.
- **Verify, don't trust.** Requesting a configuration is not evidence it took
  effect, so the `OptimizationProfilePlugin` contract requires reading the applied
  state back (`verify_engaged()`) — a profile that cannot prove it engaged does not
  get credit for the run. (The runner does not call this yet; see the status note
  under Examples.)
- **Abort, don't degrade.** When a measurement cannot be taken, the run fails
  loudly instead of silently producing a partial result that looks complete. A
  quiet fallback is worse than an error, because it ships as data.
- **No fabricated PMU codes.** If a hardware counter is not documented for the
  SKU under test, the suite does not advertise it. A plausible-looking event
  encoding guessed from a sibling microarchitecture reads as a real number and is
  indistinguishable from one downstream, which is the worst failure mode a
  benchmark has.

`docs/methodology/platform_capability_policy.md` works through how these apply to
platform and capability detection.

## Datasets

The `harness/datasets/` directory is excluded from the repository due to size. See `harness/README.md` for instructions on fetching Terminal-Bench 2 and other datasets separately.

## Running the analyzers, sweeps, and dashboards

Crisp commands for everything added recently. Activate the env first:

```bash
source ~/.cache/pypoetry/virtualenvs/agentsysperf-*-py3.12/bin/activate   # or your venv
pip install -e ".[dashboard,trace]"                                    # extras for UI + Langfuse
```

### See what's installed

```bash
agentsysperf list           # every plugin
agentsysperf analyzers list # just analyzers: cpu_bound, cache, memory_leak,
                         # memory_bandwidth, phase_profiler, scaling
```

### Per-span analyzers (cpu_bound, cache, memory_leak, memory_bandwidth)

These run automatically over a results directory — you do NOT name one; every
discovered analyzer fires. Make records, then analyze:

```bash
python examples/run_synthetic_l1_l3.py                       # writes /tmp/agentsysperf_scratch/synthetic_l1_l3/
agentsysperf analyze /tmp/agentsysperf_scratch/synthetic_l1_l3/    # tree output
agentsysperf analyze /tmp/agentsysperf_scratch/synthetic_l1_l3/ --format json
```

Note: `scaling` is a registered analyzer but does NOT fire from
`agentsysperf analyze` — it is sweep-scoped. Run it via its own driver below.

### Phase profiler (Reason/Act breakdown + inflection)

```bash
python examples/run_phase_profiler_tb2.py --dry-run   # mock LLM, no API key needed
```

### Concurrency-scaling sweep (ScalingAnalyzer: knee + bottleneck, "agents per vCPU")

The sweep runs the agent at rising concurrency, then ScalingAnalyzer finds the
saturation knee and classifies the bottleneck. It runs `analyze_sweep()`
internally and writes the verdict to SQLite.

```bash
# Dry run — SYNTHETIC modeled cells, no Docker/Harbor. Exercises the whole
# pipeline in seconds; the points are not measurements (data_source=synthetic).
agentsysperf sweep run --dry-run --output-dir /tmp/agentsysperf_sweep

# Real sweep with deterministic LLM replay (records served from a fixture).
# Quotes the trial count and asks for confirmation; -y skips the prompt.
agentsysperf sweep run \
    --fixture /path/to/your/fixture.jsonl \
    -d 0.25 -d 0.5 -d 1.0 -d 1.5 -d 2.0 -d 3.0

# Same axes, driven programmatically (equivalent; see the script for the API):
python examples/run_scaling_sweep_tb2.py --dry-run

# Read the verdict it wrote (the sweep persists to the canonical store):
python -c "
from src.storage.sqlite_store import SQLiteResultStore
s = SQLiteResultStore.open(read_only=True)
sid = s.query_sweeps()[0]['sweep_id']
v = s.query_verdicts(sid, analyzer_name='scaling')[0]
print('verdict:', v['verdict'], '| confidence:', v['confidence'])
print('knee:', v['evidence'].get('knee'))
"
```

The sweep renders in **both** dashboards: the demo app's "Concurrency Sweep"
page (Terminal-Bench / Tau-Bench sections) and the live dashboard.

The record/replay proxy can also be run standalone (record once, replay byte-identically):

```bash
python -m src.replay.proxy --mode replay --fixture <fixture.jsonl> --port 4001
python -m src.replay.proxy --mode off --port 4001    # empty LLM responses; isolate orchestration
```

### SWE-Bench experiment (5-phase pipeline)

The SWE-Bench adapter ships in core (`agentsysperf benchmarks list`) and runs
SWE-bench code repair tasks through the 5-phase agentic pipeline. It classifies
tool commands into pipeline phases:
- **Admit**: environment bootstrap (Docker container setup)
- **Retrieve**: `grep`, `find`, `cat`, `ls`, `tree` etc. (context gathering)
- **Reason**: LLM inference calls
- **Act**: `sed`, `patch`, `python`, `pip` etc. (code modification)
- **Commit**: `git diff`, `COMPLETE_TASK`, patch submission (verification)

Results are persisted to the unified store and visible in the demo dashboard
under SWE-Bench → Agent Performance. The end-to-end EMON-instrumented driver for
this experiment ships with the separate Intel-only agentsysperf-emon plugin.

### Tau-Bench experiment (5-phase pipeline)

The Tau-Bench adapter ships in core and runs Tau-Bench (tau2) customer-service
scenarios through the 5-phase pipeline. Requires tau2 installed
(`pip install tau2`). Its phase classification maps tau2's internal call
structure:
- **Admit**: session setup gap (environment + config initialization)
- **Retrieve**: user simulator LLM calls (`generate_user_message`)
- **Reason**: agent LLM inference calls (`agent_response`)
- **Act**: tool execution via `Environment.get_response`
- **Commit**: evaluation calls (`nl_assertion`, `review`)

The adapter monkey-patches tau2 internals to capture per-call latencies,
token counts, inference time, tool arguments, and response previews —
matching the recording fidelity of the standalone upstream tau-bench runner.

Results render in the demo dashboard under Tau-Bench → Agent Performance. The
end-to-end EMON-instrumented driver for this experiment ships with the separate
Intel-only agentsysperf-emon plugin.

### Density-contention experiment (experiments/scaling/)

Synthetic phase-mix agent workers with NUMA pinning under system-wide contention:

```bash
python -m experiments.scaling.run_experiment --quick     # fast smoke
python -m experiments.scaling.run_experiment --full      # full density x mix x placement matrix
```

### Dashboards (Streamlit)

```bash
streamlit run demo_app.py       --server.port 7860 --server.address 0.0.0.0   # benchmark pages: Results, Phase, Hardware Analysis, Concurrency Sweep, Density Study, Langfuse, Grafana
streamlit run live_dashboard.py --server.port 7861 --server.address 0.0.0.0   # runs real workloads on demand
```

**Pick these ports deliberately.** On a restricted lab network an inbound ACL
can be port-specific: 7860/7861 were reachable from a client browser on the
network this was developed on, while 8501 (Streamlit's default) was silently
dropped at TCP handshake — a timeout, not a refusal, so it looks like the app is
down when it is running and correctly bound to `0.0.0.0`. Confirm the bind before
blaming the network:

```bash
ss -ltnp | grep 7860                  # expect 0.0.0.0:7860, not 127.0.0.1:7860
curl -s -o /dev/null -w '%{http_code}\n' --noproxy '*' http://$(hostname -f):7860/
```

If a port is blocked, tunnel over SSH (port 22 is open) rather than hunting for
another one:

```bash
ssh -fN -L 7860:localhost:7860 <user>@<host>   # then browse http://localhost:7860
```

Note `--noproxy '*'` above: where `$http_proxy` is set, a corporate proxy may
return 403 for bare private-range addresses, which is a *separate* failure from
the port ACL. Bypass the proxy when testing so the two are not confused.

Both dashboards read the canonical store (`$AGENTSYSPERF_HOME/results.db`) through a
shared data-access layer; service URLs are derived from the host (set
`AGENTSYSPERF_HOST` to override). The two scaling experiments are distinct pages:

- **Concurrency Sweep** — agents/vCPU saturation knee + bottleneck (ScalingAnalyzer,
  SQLite `sweep_points`). Run a sweep above first.
- **Density Study** — absolute agent count × NUMA × phase-mix
  (`experiments/scaling/`, JSON `all_results.json`).

### Migrating earlier runs into the unified store

Storage was unified into a single canonical store at `$AGENTSYSPERF_HOME/results.db`
(default `~/.agentsysperf`). Earlier runs lived as scattered per-directory
`agentsysperf_results.db` + `measurement_records.json` files (e.g. under `docs/*`,
`/tmp/agentsysperf_*`, or wherever you pointed `--output`). If you have earlier
runs, consolidate them with the migration script.

It is **safe by default**: dry-run unless `--apply`, reads every source
**read-only** into a fresh store (md5-verifies originals are untouched), and
refuses to touch `harness/results/`.

```bash
# 1. DRY-RUN — scan + report what would migrate, writes nothing.
poetry run python scripts/migrate_to_unified_store.py                       # default: <repo>/docs
poetry run python scripts/migrate_to_unified_store.py --source-root /path/to/your/runs

# 2. APPLY — migrate into $AGENTSYSPERF_HOME (originals left byte-identical).
poetry run python scripts/migrate_to_unified_store.py --source-root /path/to/your/runs --apply

# 3. VERIFY — parity gate: store >= legacy per run.
poetry run python scripts/verify_backfill_parity.py
```

Notes:
- `run_id` is derived from each dir's basename (a leading `tb2_` is stripped),
  matching the old sidecar DBs and the Grafana run_id labels — so dashboards and
  Prometheus see the same ids.
- Re-running is idempotent (no duplicate rows).
- The migration **never deletes or modifies your original files** — it copies
  into the new store. Your old dirs remain a fallback.
- The legacy JSON/`/tmp` read paths in the dashboards are kept as a fallback;
  nothing has flipped to store-only yet.

### Tests

```bash
python -m pytest -q          # whole suite (analyzers, scaling, etc.)
```
