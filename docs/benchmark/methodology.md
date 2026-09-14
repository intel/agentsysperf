# Methodology

This benchmark measures how CPU silicon performs under real agentic AI
workloads, using a **deterministic record-replay** harness that eliminates
LLM API variance as a confounder.

## What the benchmark does

1. **Record** a single trajectory — one pass of the Terminal-Bench
   2.0 curated task suite running through Terminus-2 against a live LLM.
   Every LLM request and response is captured to `fixture.jsonl`, keyed
   on a canonicalized prompt hash.

2. **Replay** that trajectory deterministically across every CPU under
   test. The replay proxy serves the captured response for any matching
   `(trial_key, turn_index)` lookup. A cache miss aborts the trial loudly
   so non-determinism cannot silently leak in.

3. **Measure** wall time, p95 trial latency, CPU utilization (1-second
   samples), and per-task timestamps for every cell in the matrix.

## Why record-replay

When agentic benchmarks call live LLM APIs, run-over-run wall-time variance
is dominated by the LLM provider — typically ±15 % on the same hardware.
That noise hides the silicon signal you care about.

Record-replay eliminates the LLM as a variable: every CPU receives the
*same* responses for the *same* turn of the *same* task. Run-over-run
variance drops to ±2 %, which is small enough to make cross-CPU
comparisons defensible.

## What gets measured

| Metric | Captured per cell |
|---|---|
| Wall time | Seconds, total cell duration |
| Per-trial timing | Start / finish per trial; supports p95 latency |
| CPU utilization | 1-second samples via /proc/stat |
| Memory | 1-second samples via /proc/meminfo |
| Trial count | trials per cell × replicates |

Wall time, throughput (trials per minute), and p95 trial latency are the
primary headline metrics. CPU utilization explains *why* the headline
metrics behave the way they do.

## Workload — Terminal-Bench 2.0

[Terminal-Bench 2.0](https://github.com/laude-institute/terminal-bench) is
an open-source benchmark suite of agentic terminal tasks. The default 10
tasks span compile, test, ETL, SAT solve, interpreter, ray-trace,
compression, linear algebra, video transcode, and classical ML.

Each task runs inside a Docker container on the agent host. Terminus-2
operates the container via tmux — it runs commands, observes output,
iterates. The agent exits when it believes the task is complete or when
the per-trial timeout expires.

## Agent — Terminus-2

Terminus-2 is a tmux-based terminal agent shipped with the
[Harbor](https://github.com/laude-institute/harbor) orchestrator. It uses
a single tool — `send_keys` — to drive a terminal session.

## LLM — User's choice

The benchmark is LLM-agnostic. Record a fixture against any
OpenAI-compatible endpoint (OpenAI, Anthropic via Bedrock, local vLLM,
etc.). During replay, the proxy serves the recorded response and never
contacts the LLM.

To record your own fixture:

```bash
export OPENAI_API_KEY=sk-...
agentsysperf run -b terminal-bench --tasks task-a,task-b \
  --record my_fixture.jsonl --model gpt-4o-mini
```

## Determinism

The proxy normalizes prompts before hashing into a `trial_key` so the key
is invariant across CPUs:

- Strips Docker container hostnames (`root@a1b2c3d4e5f6:/app#`)
- Strips UUIDs and timestamps
- Strips terminal-state preamble blocks

This is what makes the same trial reproduce identically on a different
CPU — the trajectory keys match, the responses are byte-equal, only the
host CPU is different.

## Concurrency and density

Each cell runs the task suite at a fixed agent concurrency (`c`). At
`c = 5`, five containers run in parallel; at `c = 20`, twenty.
Replicates (`k`) repeat the suite to grow the trial count.

**Density** is the operating-point metric: `density = c / vCPU`. At
density > 1, the agent count exceeds vCPU count and contention starts to
show up in p95 trial latency. At density > 2, wall time floors and adding
more agents stops producing more throughput.

## Output

Results land in the canonical SQLite store (`$AGENTSYSPERF_HOME/results.db`)
and are immediately visible in the dashboard and CLI:

```bash
agentsysperf db ls              # list runs
agentsysperf db show <run_id>   # inspect measurements + verdicts
agentsysperf report <run_id>    # generate markdown/pptx report
```

The ScalingAnalyzer automatically finds the saturation knee and classifies
the bottleneck (cpu_bound, memory_bound, scheduler_oversubscription, etc.).
