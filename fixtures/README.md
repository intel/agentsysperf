# fixtures/

Replay fixtures for deterministic benchmark replay. A fixture is a recorded
sequence of LLM request/response pairs that the replay proxy serves instead
of calling a live LLM — free, deterministic, and hardware-measurement-only.

## fixtures/tb2_model_replay/

Contains a **placeholder fixture** that demonstrates the format. It is NOT a
real recorded session — every trial_key lookup will miss. To run real replays,
record your own fixture first (see below).

## How fixtures work

1. **Record** — run a benchmark with `--record` to capture LLM trajectories
2. **Replay** — run with `--replay` + `--tasks` to serve recorded responses

The replay proxy canonicalizes prompts (strips timestamps, UUIDs, container
hostnames) and matches by `(trial_key, turn_index)`. The same task produces
the same agent behavior regardless of when or where it runs — isolating
hardware measurement from LLM variance.

## Recording your own fixture

```bash
export OPENAI_API_KEY=sk-...
agentsysperf run -b terminal-bench --tasks task-a,task-b \
  --record my_fixture.jsonl --model gpt-4o-mini
```

## Replaying a recorded fixture

```bash
agentsysperf run -b terminal-bench --tasks task-a,task-b \
  --replay my_fixture.jsonl
```

## Fixture format

Each line is a JSON object with:
- `trial_key` — hash of the canonicalized first user message
- `turn` — turn index within the trial (0, 1, 2, ...)
- `response` — the full OpenAI-compatible chat completion response
- `latency_ms` — original response latency (for optional latency injection)
