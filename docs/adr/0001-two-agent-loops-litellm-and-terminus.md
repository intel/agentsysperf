# ADR 0001 — Two agent loops: LiteLLM in-process, terminus-2 in containers

- **Status:** Accepted (retroactively recorded 2026-08-01)
- **Decided:** 2026-05-27 (LiteLLM loop) / ~2026-06 (Harbor sweep)
- **Deciders:** project maintainers
- **Supersedes:** nothing. **Superseded by:** nothing.

## Why this ADR exists

The rationale for running *two different agent implementations* was real and
deliberate, but it survived only in a module docstring and a commit message. It
took a repo-wide search to reconstruct, and in the meantime the split read as
accidental fragmentation. This ADR records it so the next person does not have to
re-derive it — and flags the premises that have since gone stale.

`docs/adr/` did not exist before this file. It does now; put decisions here.

## Context

AgentSysPerf must answer two different questions about the same benchmark:

1. **Per-component attribution for one agent.** Where does a single agent's wall
   time and hardware behaviour go — inference vs tool execution vs
   provisioning? This needs sub-task spans (`admit`/`reason`/`act`/`commit`) and
   per-span L1/L3 records.
2. **Density / throughput / knee across many agents.** How many concurrent
   agents does this Xeon sustain before per-agent throughput falls?

These have incompatible measurement requirements, which is the crux.

## Decision

**Two agent paths, chosen per question.**

### 1. Single-host attribution → our own LiteLLM loop

`src/agent_loops/litellm_terminal_loop.py`, driven by
`agentsysperf run` through the `AgentInvoker` Protocol.

Grounded in `docs/methodology/DESIGN.md:427`, the "Option A" topology decision:

> AgentSysPerf drives the SUT while Terminal-Bench supplies tasks + oracle. This
> preserves per-component attribution but **requires re-implementing the TB agent
> loop against AgentSysPerf's plugin contracts — a real, budgeted cost.**

You cannot get per-turn phase spans out of a third-party agent that AgentSysPerf
does not control. The loop is a port of AgentOptimizer's `react_terminal_loop.py`
with `litellm.completion()` replacing `ExecutionEngine.generate_chat()`. Sync,
not async, because "hosted LLM latency dominates; async buys little here"
(module docstring). Introduced by commit `a0c6d549`, "Phase 5B: LiteLLM-backed
Terminal-Bench agent loop".

### 2. Concurrency sweeps → Harbor's container-per-agent path with terminus-2

`src/sweep/harbor_sweep.py`, `harness/scripts/run_density_test.sh`.

The decisive argument, from `harbor_sweep.py:21-25` and the strongest statement
of rationale anywhere in the repo:

> **Why Harbor's concurrency and not threads:** AgentSysPerf's `PsutilSampler`
> reads whole-process CPU% and would assign it to every in-process agent
> identically. Containers give each agent its own process and the node telemetry
> captures the aggregate — the quantity the sweep is actually about.

Two supporting reasons:

- **Continuity with prior work.** "the same way the colleague's AWS study ran"
  (`harbor_sweep.py:4-5`). `harness/results/density_test/` on Xeon 6787P used
  `-a terminus-2`; keeping it makes GNR↔CWF curves comparable.
- **Deliberately coarser.** "Per-span hardware detail for in-container agents is
  out of scope here... this runner deliberately works at cell granularity, which
  is the right level for density/throughput/knee."

## Consequences

**Accepted:**

- Two agents means two prompt sets, two tool vocabularies, two turn semantics.
  **Numbers from `agentsysperf run` and from a sweep are not directly
  comparable.** This is inherent, not a bug — but it must be stated wherever both
  appear, and the demo_app currently does not state it.
- Contributors see two ways to run Terminal-Bench and no signpost. That is the
  entry-point problem, tracked separately; it is a *surface* problem, not a
  reason to collapse the agent split.

**Not accepted, and now known to be wrong:**

- `harbor_sweep.py:8-9` claims the ReplayProxy makes the LLM deterministic "so
  the wall time reflects silicon, not model variance." **Measured false today:**
  `trial_key` is a sha256 of the canonicalized first user message, the committed
  fixture was recorded with a *different* agent, so every lookup misses; and
  replayed `"duration"` fields become real tmux waits (237 s for
  video-processing, 4162 s for path-tracing). The determinism this design
  depends on is absent.
- Commit `a0c6d549` flagged the confound on day one and it was never addressed:
  "this fingerprint is local Python+shell with the LLM as a network blocking
  wait; **swap to local vLLM for representative on-Xeon workload
  measurement**." Measured 2026-07-31: `reason` is 48.8 % of wall time and is
  remote AWS Bedrock. A sweep on this path measures Bedrock's queue.

## A third agent, added 2026-07-31 — and why

`harness/scripts/run_density_test_cwf.sh` uses **`-a oracle`** (harbor's built-in
agent, which runs each task's own `solution/solve.sh`).

Rationale: it is the only way to get *identical work per trial at every
concurrency level*, which is the precondition for a valid scaling measurement.
Deterministic, no LLM, no network, $0. Verified on this host: reward 1.0,
123.7 s/trial, of which `agent_execution` is 68.9 s (55.7 %) of real local CPU.

**Explicit cost:** it breaks continuity with the GNR terminus-2 curve. Oracle
numbers are a *harness-and-silicon* measurement; terminus-2 numbers are an
*agent-and-silicon* measurement. Do not plot them on one axis. `REPLAY=1`
restores the terminus-2 path when comparability matters more than determinism.

## Current agent inventory

| Path | Agent | Granularity | Comparable to |
|---|---|---|---|
| `agentsysperf run` | LiteLLM (ours) | per-span, 4 phases | itself only |
| `harbor_sweep.py` | terminus-2 | per-cell | GNR `density_test` |
| `run_density_test.sh` (GNR) | terminus-2 | per-cell | `harbor_sweep.py` |
| `run_density_test_cwf.sh` | oracle (or terminus-2 via `REPLAY=1`) | per-cell | itself; GNR only under `REPLAY=1` |
| `agentic-benchmark-4/scripts/` (external) | terminus-2 | own tree | GNR, informally |

## Rule going forward

The agent is a **recorded parameter**, never an implied consequence of which file
you ran. Every result must carry the agent name in provenance, and any view that
mixes agents must label them. Unify the *rollup*, not the agent.
