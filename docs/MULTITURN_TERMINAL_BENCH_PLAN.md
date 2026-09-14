# Terminal-Bench Multi-Turn Measurement Plan

## Overview

Terminal-Bench is already multi-turn — the ReACT agent loop executes multiple turns (thought → action → observation) until the task is complete. This plan adds **per-turn hardware counter measurement** to detect performance degradation as context grows across turns.

---

## What Already Exists

### Per-Turn Data (`TurnRecord`)
- `prompt_tokens`, `completion_tokens`
- `generation_ms` (LLM inference time)
- `command_ms` (tool execution time)
- `action`, `command`, `exit_code`

### Task-Level Data (`TrialResult`)
- `num_turns`, `num_commands`
- `total_generation_ms`, `total_command_ms`
- `total_prompt_tokens`, `total_completion_tokens`
- `wall_clock_ms`

### Per-Turn Spans (Already Implemented)
```python
# In litellm_terminal_loop.py, line 262:
span_id = f"{parent_span_id}/turn_{turn_idx}_llm"
span_ctx = track_span(ctx, span_id, kind="inference", node_id=f"llm_call_{turn_idx}")
```

L1+L3 measurements already attach to these spans — but no one has analyzed the per-turn hardware counter trends yet.

---

## What's Missing

| Gap | Why It Matters |
|-----|----------------|
| No hardware counters per turn | Can't see IPC/cache degradation across turns |
| No context size tracking | Can't correlate context growth with latency |
| No command execution spans | Tool calls not measured with L1/L3 |
| No phase separation within a turn | Can't split prefill vs decode per turn |
| No multi-turn analyzer | No automated detection of degradation curves |
| No aggregation view | Can't visualize turn-over-turn trends |

---

## Implementation Plan

### Phase 1: Enrich Per-Turn Spans (1 day)

**Goal**: Add hardware counter collection to existing per-turn spans and add command execution spans.

**Changes to `litellm_terminal_loop.py`**:

```python
def _run_turn(self, turn_idx, conversation, parent_span_id):
    record = TurnRecord(turn=turn_idx)

    # Track context size at this turn
    context_tokens = sum(
        msg.get("token_count", len(msg.get("content", "")) // 4)
        for msg in conversation
    )
    record.context_tokens = context_tokens

    # LLM inference span (already exists — L1+L3 attach here)
    span_id = f"{parent_span_id}/turn_{turn_idx}_llm"
    with track_span(ctx, span_id, kind="inference", node_id=f"llm_call_{turn_idx}"):
        response = litellm.completion(...)

    # Command execution span (NEW)
    if record.action == "shell":
        cmd_span_id = f"{parent_span_id}/turn_{turn_idx}_cmd"
        with track_span(ctx, cmd_span_id, kind="command", node_id=f"cmd_{turn_idx}"):
            result = env.exec(record.command)
        record.command_ms = ...
```

**What this gives**: L1 (CPU time, RSS) + L3 (IPC, cache miss) collected per turn for both LLM calls and command execution.

---

### Phase 2: Context Growth Tracking (0.5 day)

**Goal**: Record how context/KV-cache grows across turns.

**New fields in `TurnRecord`**:

```python
@dataclass
class TurnRecord:
    turn: int
    # ... existing fields ...
    context_tokens: int = 0           # total tokens in conversation at this turn
    cumulative_prompt_tokens: int = 0 # sum of all prompt_tokens up to this turn
    context_growth_pct: float = 0.0   # % growth since last turn
```

**In the solve loop**:

```python
for turn_idx in range(self.max_turns):
    record = self._run_turn(turn_idx, conversation, parent_span_id=task_id)
    record.context_tokens = self._count_context_tokens(conversation)
    record.cumulative_prompt_tokens = result.total_prompt_tokens
    if turn_idx > 0:
        prev_ctx = result.turns[-1].context_tokens
        record.context_growth_pct = (record.context_tokens - prev_ctx) / max(prev_ctx, 1) * 100
```

---

### Phase 3: Multi-Turn Analyzer (1 day)

**Goal**: Detect degradation patterns across turns using per-turn hardware data.

```python
class MultiTurnAnalyzer:
    """Detects performance degradation patterns across turns.

    Patterns detected:
    - context_growth_cliff: TTFT jumps when context exceeds L3
    - kv_cache_spill: IPC drops / cache miss rises mid-conversation
    - tool_dominance: Tool calls > 70% of total time (not LLM-bound)
    - linear_scaling: Healthy — latency grows proportionally with context
    - superlinear_scaling: Unhealthy — quadratic attention cost
    - late_turn_memory_bound: Early turns compute-bound, late turns memory-bound
    """

    name = "multi_turn"
    input_layers = frozenset(["l1", "l3"])

    def analyze(self, records, *, context=None):
        # Group spans by task, order by turn number
        # Compare IPC/cache_miss at turn 1 vs turn N
        # Detect inflection point where metrics degrade
        # Map to solutions (RAG, KV compression, spec decode)
        ...
```

**Inflection detection logic**:

```python
def _find_inflection(self, turn_metrics: list) -> Optional[int]:
    """Find the turn where IPC drops below threshold or cache miss spikes."""
    for i in range(1, len(turn_metrics)):
        ipc_now = turn_metrics[i]["ipc"]
        ipc_prev = turn_metrics[i-1]["ipc"]
        cache_now = turn_metrics[i]["cache_miss_pct"]

        # IPC dropped > 30% in one turn
        if ipc_prev > 0 and (ipc_prev - ipc_now) / ipc_prev > 0.3:
            return i

        # Cache miss jumped above L3 budget threshold
        if cache_now > 60 and turn_metrics[i-1]["cache_miss_pct"] < 40:
            return i

    return None  # No cliff detected
```

**Solution mapping per turn range**:

```python
def _map_solutions_by_phase(self, inflection_turn, total_turns):
    """Different solutions apply at different turn ranges."""
    return {
        "pre_inflection": {
            "turns": f"1-{inflection_turn - 1}",
            "verdict": "compute_efficient",
            "solutions": ["No memory optimization needed"],
        },
        "at_inflection": {
            "turns": f"{inflection_turn}",
            "verdict": "working_set_overflow",
            "solutions": ["NUMA-aware scheduling", "Data tiling", "Cache partitioning"],
        },
        "post_inflection": {
            "turns": f"{inflection_turn + 1}-{total_turns}",
            "verdict": "weight_streaming / kv_cache_pressure",
            "solutions": ["Speculative decoding", "RAG (reduce context)", "KV-cache compression"],
        },
    }
```

---

### Phase 4: Live Dashboard Multi-Turn Tab (0.5 day)

**Goal**: Visualize per-turn metrics in the Streamlit dashboard.

```python
# New tab: "Multi-Turn Analysis"

# Line chart: IPC per turn (should degrade)
fig = go.Figure()
fig.add_trace(go.Scatter(x=turn_numbers, y=ipc_per_turn, name="IPC"))
fig.add_trace(go.Scatter(x=turn_numbers, y=cache_miss_per_turn, name="Cache Miss %", yaxis="y2"))

# Vertical line at inflection point
fig.add_vline(x=inflection_turn, line_dash="dash",
              annotation_text="L3 budget exceeded")

# Context size vs TTFT scatter
fig2 = go.Figure(go.Scatter(x=context_tokens_per_turn, y=ttft_per_turn, mode="markers"))
fig2.update_layout(title="Context Growth vs Latency")

# Per-turn breakdown table
st.dataframe(turn_records_with_hardware_counters)
```

---

### Phase 5: Integration with MemoryBandwidthAnalyzer (0.5 day)

**Goal**: Feed per-turn data into the existing analyzer to get per-turn solution mapping.

```python
# For each turn, run MemoryBandwidthAnalyzer
for turn_idx, turn_records in enumerate(per_turn_records):
    results = analyzer.analyze(turn_records)
    for r in results:
        if r.verdict != "no_memory_bottleneck":
            print(f"Turn {turn_idx}: {r.verdict}")
            # Turn 1: no bottleneck
            # Turn 3: working_set_overflow → NUMA scheduling
            # Turn 5: weight_streaming → spec decode / RAG
```

---

## Expected Demo Output

```
Terminal-Bench Task: "Set up a Python web server with database"
Turns: 8 | Commands: 12 | Total: 45.2s

Turn  Context   TTFT    IPC   Cache%  Verdict
─────────────────────────────────────────────────────
  1     512    15ms   3.40    8%     compute_efficient
  2    1024    28ms   3.10   15%     compute_efficient
  3    2048    52ms   2.60   32%     working_set_overflow
  4    3072    95ms   2.10   48%     working_set_overflow
  5    4096   180ms   1.50   65%     ← L3 BUDGET CROSSED
  6    5120   320ms   1.10   78%     weight_streaming
  7    6144   480ms   0.90   85%     weight_streaming
  8    7168   650ms   0.75   91%     weight_streaming

Inflection: Turn 5 (context=4096 tokens exceeded L3 budget of 120MB)

Solutions applicable from Turn 5 onward:
  → RAG: Retrieve relevant context instead of accumulating full history
  → KV-Cache Compression: Reduce cache footprint to stay in L3
  → Speculative Decoding: Amortize weight loads in memory-bound turns
```

---

## What This Proves

1. **Single-turn benchmarks are misleading** — They only show Turn 1 performance
2. **The cliff is measurable** — Hardware counters pinpoint exactly where degradation starts
3. **Solutions are turn-dependent** — Early turns need different optimization than late turns
4. **RAG is a memory optimization** — Not just accuracy; it keeps context below L3 budget
5. **Terminal-Bench is a real multi-turn workload** — Not synthetic; agent naturally accumulates context

---

## Timeline

| Phase | What | Effort | Deliverable |
|-------|------|--------|-------------|
| 1 | Enrich per-turn spans (L1+L3 + command spans) | 1 day | Per-turn hardware counters |
| 2 | Context growth tracking | 0.5 day | Context tokens per turn field |
| 3 | MultiTurnAnalyzer | 1 day | Degradation detection + inflection point |
| 4 | Dashboard multi-turn tab | 0.5 day | Visualization of turn curves |
| 5 | Integration with MemoryBandwidthAnalyzer | 0.5 day | Per-turn solution mapping |

**Total: ~3.5 days**

---

## Dependencies

- **Phase 1 requires**: A running LLM endpoint (vLLM or LiteLLM-compatible) to generate real multi-turn data with hardware counters
- **Phases 2-5**: Can be developed with synthetic/mock turn data first, validated with real LLM later
- **Hardware**: perf_event_paranoid=0 (already set on this system)

---

## Connection to Existing Components

| Component | Role in Multi-Turn |
|-----------|-------------------|
| `track_span` (runner.py) | Already supports nesting — per-turn spans work today |
| `L1SubSpanMeasurement` | Captures CPU time, RSS per turn (attaches to span) |
| `L3PerfMeasurement` | Captures IPC, cache miss per turn (attaches to span) |
| `MemoryBandwidthAnalyzer` | Run per-turn to get solution mapping at each stage |
| `PlatformInfo` | Provides L3 budget threshold for cliff detection |
| `PrometheusExporter` | Export per-turn metrics for Grafana time-series |
| Live Dashboard | Visualize degradation curves |
