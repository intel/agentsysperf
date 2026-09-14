# Phase Characterization — Benchmark Coverage Analysis

## Which Benchmarks Cover Which Phases?

| Phase | Terminal-Bench | SWE-Bench | Tau-Bench | Notes |
|-------|--------------|-----------|-----------|-------|
| 01 Admit (auth/policy/route) | — | — | — | No benchmark has a governance layer today |
| 02 Retrieve (vector/rerank/fetch) | Weak (history only) | Codebase search (grep, find, file reads) | Product catalog lookup, user history queries | Tau-Bench strongest |
| 03 Reason (LLM inference) | Excellent | Excellent | Excellent (agent + user sim) | All benchmarks cover this |
| 04 Act (tool/API/code exec) | Excellent (shell exec) | Good (shell: git diff, patch) | Excellent (API: get_order, cancel_flight) | All benchmarks cover this |
| 05 Commit (write-back/audit/cache) | Minimal | Patch submission only | State mutations (update_order, modify_booking) | Tau-Bench strongest |

---

## Terminal-Bench: What It Covers Today

The `litellm_terminal_loop.py` ReACT loop has this structure per turn:

```
Turn N:
  [Phase 03 - Reason]   litellm.completion() → track_span(kind="inference")
  [Phase 04 - Act]      env.exec(command)    → track_span(kind="execution")
```

**Already instrumented with L1+L3 hardware counters per span.**

What's missing:
- Phase 01 (Admit): No auth. Goes straight to inference.
- Phase 02 (Retrieve): Conversation history grows but no external retrieval.
- Phase 05 (Commit): StepTrace persistence is post-task, not per-turn.

### Hardware Characterization Possible Today (Terminal-Bench)

| Phase | Measurable? | What You Get |
|-------|------------|--------------|
| 03 Reason | Yes | IPC, cache miss per LLM call. Shows weight_streaming pattern. |
| 04 Act | Yes | IPC, cache miss per shell command. Bimodal: I/O wait vs compute. |
| 03 vs 04 trend | Yes | Per-turn: how the ratio shifts as context grows. |
| Inflection detection | Partial | Can detect when Act cumulative time > Reason time. |

---

## What Each Benchmark Uniquely Adds

### Tau-Bench (Recommended for Full 5-Phase)

- **Phase 02**: Natural retrieval — user asks about an order, agent queries product catalog, checks order history. This is real vector/DB lookup behavior.
- **Phase 04**: Rich tool variety — read-only lookups (get_product, get_order) vs mutating APIs (cancel_order, update_booking). Different hardware profiles.
- **Phase 05**: Real commit — state mutations to the environment (order status changes, booking modifications). Actual write-back.
- **Multi-turn**: 5–15 turns per task. Shows accumulation effects clearly.
- **Dual LLM**: Agent model + user simulator model — two inference calls per turn. Unique characterization opportunity.

### SWE-Bench

- **Phase 02**: Codebase search (grep, find, cat). I/O-heavy retrieval with variable result sizes.
- **Phase 04**: Shell execution with subprocess overhead (git, pytest, build tools). Often compute-heavy.
- **Phase 05**: Only the final patch submission. Not per-turn.
- **Long-running**: 10–50+ turns. Best for showing late-turn degradation.

### LangChain Web-Search

- **Phase 02**: Strongest retrieval — actual web fetch + HTML parse + LexRank summarization. CPU-intensive retrieval.
- **Phase 04**: Minimal (no tool beyond search).
- **Batch scaling**: Shows contention at 32–64 concurrent requests. Unique for NUMA/bandwidth analysis.

---

## Practical Implementation Path

### Step 1: Terminal-Bench Phase Tags (Now)

Add `phase` attribute to existing spans in `litellm_terminal_loop.py`:
- `track_span(..., phase="reason")` for LLM inference
- `track_span(..., phase="act")` for command execution

This gives immediate per-phase hardware characterization for the two dominant phases.

### Step 2: Tau-Bench Phase Instrumentation (1–2 days)

Instrument the Tau-Bench adapter with full 5-phase spans:
- Phase 02: Span around environment.get_response() when tool is a lookup
- Phase 03: Span around generate() for agent LLM calls
- Phase 04: Span around tool execution (mutating operations)
- Phase 05: Span around state commit after mutations

### Step 3: Phase 01 Simulation (Optional, 0.5 day)

Add a lightweight policy-check middleware to the adapter:
- Token validation (simulated)
- Rate limiting check
- Route selection

Purpose: Prove it's negligible (<1% of wall-clock) with hardware data.

---

## Expected Results by Benchmark

### Terminal-Bench (12 turns, "set up Python web server")

```
Phase     Wall%   CPU%    IPC    Cache Miss   Pattern
──────────────────────────────────────────────────────────
Reason    72%     55%     0.9    91%          weight_streaming
Act       28%     45%     3.2    12%          compute_efficient (mostly)
```

### Tau-Bench (8 turns, retail order inquiry)

```
Phase     Wall%   CPU%    IPC    Cache Miss   Pattern
──────────────────────────────────────────────────────────
Admit      0.3%   0.2%   4.5    3%           compute_efficient
Retrieve  18%     22%    2.1    42%          working_set_overflow
Reason    52%     38%    0.9    91%          weight_streaming
Act       24%     32%    2.8    18%          mixed (API I/O + compute)
Commit     5.7%   7.8%   3.4    8%           I/O_dominant
```

### SWE-Bench (25 turns, fix Django bug)

```
Phase     Wall%   CPU%    IPC    Cache Miss   Pattern
──────────────────────────────────────────────────────────
Retrieve  15%     20%    2.4    35%          I/O + working_set
Reason    60%     42%    0.9    92%          weight_streaming
Act       25%     38%    2.6    22%          subprocess_compute
```

---

## Key Insight for Demo

Terminal-Bench proves the **Reason vs Act** split — the two biggest phases.
Tau-Bench proves the **full pipeline** and shows Retrieve/Commit are non-trivial at scale.
Together they validate the PPT thesis: "Four of five phases are CPU-resident."

**Start with Terminal-Bench** because it's working today with hardware counters.
**Graduate to Tau-Bench** for the complete story.
