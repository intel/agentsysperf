# INCIDENT: agent commands ran on the HOST, polluting the repo venv

**Date:** 2026-06-03 · **Run:** `mteb_emr` (mteb-retrieve on EMR) · **Severity:** high

## What happened
A real TB2 run of `mteb-retrieve` built and started a Harbor container and
captured L1/L3/PerfSpect + 61 step traces — but the agent's shell commands
executed on the **host**, not in the container. The agent, trying to fix the
task's broken `mteb` import, ran `pip install` commands that mutated the
**repository's Poetry venv**:
- installed `mteb==1.36.8` (not in lock)
- `pip install --upgrade transformers` → 5.9.0 → 5.10.1
- pulled `datasets` to 4.8.5, bumped `rich`
- the agent also `ls`/`cat`'d real host paths (`/home/<user>/Projects/agentsysperf/...`)

Evidence it ran on host: `cat /app/data.txt` **errored** (no such file on host)
while `ls /home/<user>/Projects/agentsysperf/docs` **succeeded**. `/app/data.txt`
exists only inside the container.

## Root cause
`TerminalBenchAdapter.run_task()` (src/benchmarks/terminal_bench/adapter.py:144)
creates the environment via `_make_environment()` but **never calls
`env.start()`**. So during a real run the Harbor container is never provisioned.
The agent loop's `SyncEnvironmentWrapper.exec()` then runs commands — and instead
of failing closed, they executed against the host shell/Python.

(My earlier manual probes called `start()` explicitly, which is why they ran
in-container and looked correct — masking this bug.)

## Two defects to fix (before any further real runs)
1. **Adapter never starts/stops the container.** `run_task` must
   `await env.start()` (force_build=True) before driving the agent and
   `env.stop(delete=True)` in teardown. The Harbor adapter's start/stop
   signatures were already fixed for Harbor 0.8.0 in this session.
2. **exec must fail closed, never fall through to host.** If the environment
   isn't started or its exec errors, the loop must record an error turn — NOT
   execute on the host. A Harbor `exec` on an unstarted env should raise and the
   loop should treat it as a failed command, not run it locally.

## Remediation done
- Removed agent-installed `mteb`; restored `transformers==5.9.0` (locked);
  reinstalled `datasets` (legitimate harbor dep). `pip check` now shows only the
  pre-existing pyproject/lock drift (`rich`, `packaging`) unrelated to the agent.
- Verified: core + harbor + trace imports OK; all 10 trace tests pass.

## Data validity
The `mteb_emr` run's measurements/traces are **host-process artifacts, not
in-container** — discard for hardware-characterization purposes. The run is
still useful as the evidence that exposed the bug.

## FIX (2026-06-03, verified)
Root cause was actually THREE linked defects:
1. **`_make_environment` looked at `extra["harbor"]`** but `list_tasks()` stashes
   the source task under `extra["raw"]`, so the harbor handle lived at
   `extra["raw"].extra["harbor"]` → lookup always missed → **always fell back to
   `StandaloneEnvironment` (host subprocess)**. Fixed to also check the nested
   location.
2. **`run_task` never started the env.** Added `_start_environment()` (drives the
   backend's async `start()`) before setup, and `_teardown_environment()` now
   calls the async `stop()` (removes the Harbor container/image).
3. **exec did not fail closed.** The agent loop now catches exec errors, records
   an error turn + feeds the agent an observation, and never proceeds — so a
   broken env can't silently run on the host.
Also: `session_id` (= task id like `terminal-bench/mteb-retrieve`) contained a
`/` that broke `mkdtemp`/docker project naming — sanitized to a flat token.

**Verification:** via the real `run_task` path, `/app/data.txt` (29 lines) is
visible inside the container, `whoami`=root, and `/home/<user>` is NOT
accessible from the container (`isolated`). Clean teardown, no leftover
containers. 10 trace tests pass.

## Guardrail recommendation
Run benchmarks against a container/sandbox that the harness *guarantees* is
isolated; never let a not-started env silently degrade to host exec. Consider
running the whole benchmark as a non-repo user or in a throwaway venv so a
future escape can't touch the project environment.
