# Changelog

## 1.0.1

A maintenance release of 1.0.0: bug fixes, refreshed dependency pins and
documentation corrections. The plugin protocols and the `agentsysperf`
commands are unchanged.

### Fixed

- **Concurrent runs** (`run-streams --slots N`): the node-wide layers
  (`l1_system`, `l3_perf`, `perfspect`) are kept out of concurrent slots. Each
  slot used to store the whole node's activity, every slot's work, as its own
  task's row, and the L3 analyzers published per-task verdicts from it.
- **One PMU owner per run:** `l3_perf` and `perfspect` no longer program the
  hardware counters at the same time. With both on, the counters multiplexed
  (each event enabled 41-89% of the time), and the scaled IPC and miss rows
  looked valid but moved from run to run.
- **PMU contention warning:** the check matched the other workers' own
  `pgrep` command lines, so every `run-streams` worker warned of contention
  with no `perf` running. It now matches the `perf` executable, and runs only
  when the run registered a node-wide layer.
- **`run-streams` proxy ports:** the port check binds with `SO_REUSEADDR`, as
  the replay proxy does, so a port the previous run left in `TIME_WAIT` is no
  longer refused. A live listener still is.
- **Live dashboard** (`live_dashboard.py`): no longer stops with
  `NameError: name '_TMP' is not defined` as soon as a benchmark run has
  produced results.
- **Demo dashboard** (`demo_app.py`): the Synth Density Study diagram names the
  detected platform instead of always reading "Detected Platform".
- **Replay proxy**: `python -m src.replay.proxy --mode record` without
  `--fixture` exits with a usage error (status 2). It used to start, pass its
  health check and fail every call it was asked to record.
- **Harness scripts**: container cleanup removes only the harbor trial
  containers of the tasks the script runs. It used to force-remove every Docker
  Compose container on the host, other users' included (and, in
  `numa_tb2_comparison.sh`, any container with "harbor" in its name).
- **Harness scripts**: `run_density_test.sh`, and `run_density_test_cwf.sh`
  with `REPLAY=1`, require `FIXTURE` instead of defaulting to a path that does
  not exist; `harness/config/local.yml` points at the shipped placeholder
  fixture.

### Changed

- **Harbor telemetry is off by default.** Every `harbor run` that AgentSysPerf
  launches (`agentsysperf sweep run` and the harness scripts) sets
  `HARBOR_TELEMETRY=0`, unless `HARBOR_TELEMETRY` is already set in your
  environment. `export HARBOR_TELEMETRY=1` opts back in.
- **Langfuse compose** (`monitoring/langfuse/docker-compose.yml`):
  `LANGFUSE_NEXTAUTH_SECRET` and `LANGFUSE_SALT` are required and have no
  built-in defaults, and Postgres is pinned to `postgres:17-alpine`. The file
  now states that a complete Langfuse v3 deployment also needs ClickHouse,
  Redis, S3-compatible storage and the Langfuse worker, which it does not run.

### Security

- Refreshed dependency pins in `requirements.txt`: aiohttp 3.14.5, certifi
  2026.10.23, filelock 4.2.4, litellm 1.100.4, multidict 7.1.0, pillow 13.0.0,
  pyjwt 2.15.1, streamlit 1.66.1, urllib3 2.8.0 and yarl 1.25.1. This clears
  the 81 advisories known on 2026-11-04 against the 1.0.0 pins;
  `pip-audit -r requirements.txt` reports none. CVE-2026-114612 in litellm is in
  the LiteLLM proxy server, which AgentSysPerf does not run; litellm moves to a
  fixed version anyway.
- If you installed 1.0.0 from `requirements.txt`, reinstall from the 1.0.1 file
  (`pip install -r requirements.txt`): updating the source does not update the
  packages already in your environment.

### Documentation

- `docs/GETTING_STARTED.md`: clone from https://github.com/intel/agentsysperf;
  `agentsysperf db show` takes an exact run id; record/replay examples pass
  `--tasks`, which `--replay` requires; troubleshooting entries that described
  messages and options that do not exist are corrected or removed; the
  phase-tagging `track_span` example passes its `RunContext`.
- `docs/PLUGIN_DEVELOPMENT.md`: every example now matches the plugin protocols
  in `src/protocols.py` (the Measurement lifecycle, `MeasurementRecord` fields,
  benchmark adapter and optimization profile signatures), and the test suite
  runs them.
- `README.md`, `docs/README.md` and the task-sized replay example: the fixtures
  description is corrected, links to missing documents are removed, commands
  use the plain virtualenv instead of `poetry run`, the harbor telemetry
  default is documented, and the slot sweep takes `--slots-values 12,24,48,96`.

### Upgrade notes

- Langfuse compose: export `LANGFUSE_NEXTAUTH_SECRET` and `LANGFUSE_SALT`
  before `docker compose up` (`openssl rand -base64 32` for each), and keep
  them stable: SALT hashes the stored API keys. If you kept the old public
  defaults, the new values invalidate existing sessions and API keys. An
  existing `langfuse_pg_data` volume initialised by a PostgreSQL major other
  than 17 must be migrated (dump and restore) or recreated.
- Scripts that started the replay proxy in record mode without `--fixture`
  now stop at startup; pass the fixture path to record into.
