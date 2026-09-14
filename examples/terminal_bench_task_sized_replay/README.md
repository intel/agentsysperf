# Task-Sized Terminal-Bench Replay

This example shows how to run Terminal-Bench tasks in parallel with
task-sized Docker containers using the `runc` runtime. Each task keeps
its declared CPU size and memory limit. You supply your own recorded
replay fixture (see "Recording a fixture" below).

## Prerequisites

- Python `>=3.12,<3.15` (Python 3.12 through 3.14).
- Linux on x86_64.
- Docker configured with the `runc` runtime.
- Harbor installed and able to resolve the Terminal-Bench dataset.
- A recorded replay fixture (record your own with `--record`).

## Recording a fixture

```bash
export OPENAI_API_KEY=sk-...
agentsysperf run -b terminal-bench --tasks task-a,task-b \
  --record my_fixture.jsonl --model gpt-4o-mini
```

## Running with replay

```bash
agentsysperf run-streams \
  --slots 48 \
  --stream-multiple 2 \
  --replay /path/to/your/fixture.jsonl \
  --ref 1 \
  --run-id task_sized_s48
```

The scheduler resolves task metadata and pre-pulls task images before any
measured container starts. It creates one queue per task CPU-size class and
assigns tasks only to matching slots.

When `--tasks` is omitted, the default 23-task selection from
`clean_tasks_23.txt` is used. Override with a comma-separated list to
select specific tasks.

## Sweep across slot counts

```bash
python examples/terminal_bench_task_sized_replay/sweep_total_cores.py \
  --replay /path/to/your/fixture.jsonl \
  --slots 12 24 48 96
```
