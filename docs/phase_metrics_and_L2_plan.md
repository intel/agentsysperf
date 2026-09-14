# Plan: Phase-Categorized Metrics/Timeline (Feature A) + L2 Flame Graphs (Feature B)

**Status:** proposal · **Date:** 2026-06-05
**Inputs:** 3 expert agents (Xeon-perf, agentic-observability, adversarial sanity check).
The sanity review materially changed this plan — both features are de-scoped from
the raw expert proposals. Read the "Why" notes; they're load-bearing.

## The fact that reframes both features
**Inference is a remote network wait.** `litellm.completion()` blocks the Python
process on a socket; the model runs elsewhere. Therefore any *hardware* metric
(IPC, cache, TMA, flame graph) attributed to the **inference** phase measures the
Python interpreter idling on `recv()` — not the model. Consequences:
- Per-phase **hardware** is only meaningful for **execution** (container shell work)
  and **orchestration** (prompt assembly, parsing in the loop).
- For **inference**, expose only **tokens / cost / latency**. Suppress IPC/TMA/flame.
This both improves correctness and *removes* work.

---

# Feature A — Phase metrics + timeline  → GO (de-scoped)

## A0 (STEP 0, prerequisite for everything): wire StepTrace into the main run path
`store_spans()` is currently called only in `scripts/run_tb2_dashboard_demo.py`,
NOT in the production path (`terminal_bench/adapter.py` + `litellm_terminal_loop.py`).
A1's token rollup and A2's timeline both read `query_spans` → today they'd read an
empty table.
- Call `turns_to_steptraces()` → `store_spans()` from the real run path (after the
  task span closes, not interleaved with measurement).
- Test: assert rows land for a NON-demo run, and span_ids exactly match the
  hardware sub-span convention `{task}/turn_{i}_llm` / `{task}/turn_{i}_cmd`
  (the only thing joining hardware-JSON to tokens-SQLite — orphans if it drifts).

## A0b (STEP 1): one window-attribution source of truth — absolute timestamps
StepTrace `start_ts_us`/`end_ts_us` are currently 0 (unwired). Thread
`time.time_ns()` (epoch, NOT monotonic — the schema promises epoch-µs for
Perfetto/OTel) at the existing `gen_t0`/`cmd_t0` stamp sites:
`SpanRecord` → add fields to `TurnRecord` → set in `turns_to_steptraces` → existing
SQLite columns (no schema change).
- This single change unblocks: A2's Gantt, StepTrace timestamps, and any future
  L2/Tempo per-phase windows. Do NOT build a second window-merge mechanism.

## A1: per-phase metrics (SQLite-backed, NOT Prometheus)
- Extend `BreakdownAnalyzer`: `input_layers={l1,l3,perfspect}`; roll up hardware by
  kind for **execution + orchestration only**. Use **instruction-weighted** IPC/cache
  (L3 payload has instructions/cycles), never naive mean.
- **Inference rollup = tokens_in/out, cost_usd, llm_call_count, generation_ms** from
  the SQLite `spans` table. NO hardware for inference.
- **Do NOT emit `agentsysperf_phase_*` to Prometheus.** A finished-run categorical
  rollup is not live time-series; Prometheus is the wrong store and the cost-copy
  duplicates the SQLite source of truth. Render from SQLite into the PPTX report
  (and Langfuse already shows per-step tokens/cost).
  - *Why (sanity review):* avoids the two-stores-one-join-key fragility the plan
    itself flagged; keeps SQLite authoritative for cost.

## A2: phase timeline visualization
- **Per-task timeline:** one matplotlib **Gantt** slide in the existing
  `XeonPowerPointGenerator` (it already does matplotlib→PNG→slide), reading
  `query_spans`: rows = turns, x = time from `start_ts_us`, color = phase, draw
  orchestration as the inter-span **gaps** within a task (residual made visible).
  Requires A0b.
- **Interactive waterfall:** lean on **Langfuse** — already renders turn-by-turn
  waterfall with durations/tokens/cost for free (LiteLLM callback). No new infra.
- **Run-level phase distribution (stacked bars across tasks):** ALREADY EXISTS in
  `xeon_pptx.py::_add_breakdown_chart`; A1's richer rollup just enriches it.
- **Do NOT build a Grafana state-timeline** (Prometheus is the wrong tool for
  variable-duration discrete spans). Reserve the Grafana **Tempo** waterfall for
  Phase 3 (task #53) — A0b's timestamp fix unblocks it later.

## Feature A risks
- Inference hardware suppression must be explicit in code + report, or a reader
  sees "inference IPC" and misreads it.
- `store_spans` writes from the hot path — ensure it's post-span-close; cost calc
  (`litellm.cost_per_token`) runs per row (fine offline).
- Span_id naming is the sole hardware↔token join key — add the orphan test.

---

# Feature B — L2 flame graphs  → NO-GO as specified; DEFER

The host-only continuous-`perf record` MVP fails three independent ways, any one
fatal:
1. **Container symbolization:** execution runs in a Harbor container that `stop()`
   DELETES before post-processing → the entire execution phase resolves to
   `[unknown]`. The interesting phase is exactly the blind one.
2. **Shared SLURM node:** `perf record -a` captures neighbors' jobs; every graph is
   contaminated. Fixing with `--cgroup` contradicts the "system-wide sees the
   container" rationale and is real work, not a flag.
3. **Wrong output for the audience:** SKU selection / CPU-arch decisions are
   answered by **TMA** (what the CPU is doing — already have it via PerfSpect/L3).
   Flame graphs answer "which functions" — a *software-optimization* question for a
   different audience. An Intel architect reads TMA, not a flame graph.

Also: per-span slicing of a 99 Hz system-wide capture is statistically empty for
2–4s spans (below the plan's own 50-sample gate) and clock-fragile (needs
`perf record --clockid CLOCK_MONOTONIC`, which the original plan omitted).

**Decision:** cut L2 from the near-term roadmap. Revisit ONLY as a targeted,
opt-in, **execution-phase-only**, properly **cgroup-scoped**, `--clockid
CLOCK_MONOTONIC`, **symbolize-before-teardown** (snapshot container symbols while
it's alive), **per-phase (never per-span)** deep-dive — and only after per-phase
TMA is in stakeholders' hands and a concrete "why is execution IPC bad?" question
actually arises.

---

# De-risked execution order (the whole near-term plan)
0. Wire `turns_to_steptraces`+`store_spans` into the real run path. Test: rows for
   a non-demo run; span_ids match hardware convention.
1. Thread `time.time_ns()` into `TurnRecord` → builder → `start_ts_us`/`end_ts_us`.
   Verify: nonzero, monotonically sane epoch timestamps.
2. Extend `BreakdownAnalyzer`: execution+orchestration hardware rollup (instruction-
   weighted); inference = tokens/cost/latency only.
3. One matplotlib Gantt slide in `XeonPowerPointGenerator` (reads `query_spans`);
   Langfuse covers the interactive waterfall.
4. STOP (MVP done). L2 only if a concrete execution-IPC question survives, scoped
   as above.

# Cut from MVP (over-engineering)
- All near-term L2.
- `agentsysperf_phase_*` Prometheus metrics + cost copy (render from SQLite instead).
- Instruction-weighting for inference (moot once inference hardware suppressed).
- Tempo/eBPF/off-CPU (already Phase 3; don't leak forward).
