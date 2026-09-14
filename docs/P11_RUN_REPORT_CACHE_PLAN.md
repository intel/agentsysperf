# P11 — `agentsysperf run` / `report` + replay-journal caching

Folds the benchmark runner + report generation into the CLI (the deferred P9
items) and wires the EXISTING record/replay journal for the cost speedup.
Builds on the storage rehaul (P0–P9): `RunContext(result_store=open())` +
`persist_run`, the `db` subcommands, the DAL the dashboards read.

## Decisions (user, 2026-06-11)
- **Caching = LLM-replay journal ONLY.** Use the existing `ReplayProxy`/fixture
  system. NO task-result skip cache (it would reuse stale hardware measurements
  — a correctness trap for a *hardware* benchmark).
- **Reports: pptx + NEW markdown.**

## Why replay-journal is the right cache (and the MLCommons "journal" answer)
A run produces TWO things: the task *outcome* (pass/fail, tokens — depends on
task+model+agent) and the *hardware measurements* (IPC/EMON — depend on the
CPU). The replay journal records the LLM turns once (`--record`) and serves them
byte-identical thereafter (`--replay`), so re-runs:
- **skip** the dominant cost — LLM latency + $ + nondeterminism (the dramatic
  speedup; run-over-run variance drops ~±15% → ~±2%), but
- **still execute** the agent's shell commands and **still measure hardware** on
  the current silicon — so the numbers remain honest.

This is almost certainly the MLCommons speedup the user remembers (CM/CK
content-addressable result caching). The fixture is keyed by
`sha256(canonicalized first user message)` (host/timestamp/tmp stripped) so the
SAME task replays identically across machines — the cross-host property a
benchmark needs. Infra already exists: `src/replay/{proxy,manager,fixture}.py`.

## Scope

### 1. `agentsysperf run` (cli.py) — fold the canonical runner
Built on `scripts/run_tb2_dashboard_demo.py` (the most complete: by-name select,
per-task timeout, error isolation, incremental save) + the P5 store path.

```
agentsysperf run [--benchmark terminal-bench]
  (task selection — maps to list_tasks(include=/limit=))
  --num-tasks N          # first N tasks
  --full                 # all tasks in the dataset
  --tasks a,b,c          # explicit names (include=)
  (agent)
  --model gpt-4o-mini  --max-turns 30  --timeout 420
  (replay journal — the cache)
  --record <fixture.jsonl>   # run live, capture LLM turns to fixture
  --replay <fixture.jsonl>   # serve from fixture (no live LLM, no $)
  (output)
  --run-id <id>          # default: benchmark + timestamp
  --output <dir>         # artifact dir (default $AGENTSYSPERF_HOME/artifacts/<run_id>)
```
- Wraps the run in `ReplayProxy(mode=..., fixture=...)` when `--record/--replay`
  set; merges `proxy.env` into `os.environ` so litellm targets the proxy
  (litellm reads OPENAI_API_BASE/OPENAI_BASE_URL from env — verified, no invoker
  change). No replay flag → live OpenAI as today (`OPENAI_API_KEY`).
- `RunContext(run_id, measurements=discover(), result_store=open(), run_metadata={benchmark_id, model, hardware_sku=detect_platform(), optimization_profile, owner_id=$USER, host_id=hostname})`.
- Per task: `track_span` → `adapter.run_task(invoker)` under a thread timeout
  (lifted from dashboard_demo); `persist_run` at stop (dual-write JSON+store).
- `--replay` validates the fixture first (`validate_fixture`) — fail loud.

### 2. `agentsysperf report` (cli.py) — fold report gen onto the protocol
```
agentsysperf report <run_id> [--format pptx|md] [--out <path>]
```
- `--format pptx` → `XeonPowerPointGenerator().generate_report(run_id, store=open(read_only), output_path)`.
- `--format md`  → NEW `MarkdownReportGenerator` (below).
- Reads via the store (`open(read_only=True)`); errors clean if run_id absent.

### 3. NEW `MarkdownReportGenerator` (src/reporting/markdown_report.py)
- Implements the `ReportGenerator` protocol: `name="markdown"`,
  `output_formats=frozenset(["md"])`, `generate_report(*, run_id, store, output_path)`.
- Consumes the SAME store API as the pptx generator (`get_run`, `query_tasks`,
  `query_verdicts`, `query_measurements`): a run summary, per-task table
  (pass/duration/IPC/cache), analyzer verdicts, hardware fingerprint.
- Registers via the `agentsysperf.report_generators` entry point (pptx already does).

## Out of scope (explicitly NOT doing)
- Task-result skip cache / `--use-cache` / `--resume` (user declined — stale-HW risk).
- HTML report (markdown only this phase).
- `agentsysperf serve` / `migrate` (P10 territory).

## Verification
- `--record` then `--replay` of the SAME 2-task run → replay run does ZERO live
  LLM calls (no OPENAI_API_KEY needed), produces task results + hardware
  measurements; `db show` lists both runs.
- `agentsysperf run --num-tasks 2 --replay <fix>` lands a run readable by `db ls`/
  `db show` and renders in demo_app (via the P6 DAL) with no code change.
- `agentsysperf report <run> --format md` writes a markdown file with the run's
  tasks/verdicts; `--format pptx` still works.
- Live-key smoke (user-provided key): `agentsysperf run --num-tasks 1 --record fix`
  completes one real TB2 task end-to-end.

## Testing notes
- Replay/report paths are unit-testable WITHOUT a key (replay needs no LLM; a
  tiny hand-authored fixture + the synthetic adapter, or a mocked invoker).
- The ONE live-key step (`--record` against real TB2) needs the user's key +
  network + Docker/Harbor — run interactively, not in the test suite.
