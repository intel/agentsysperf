# Multi-Dimensional Breakdown Analysis

AgentSysPerf's breakdown analysis provides time/resource attribution across three dimensions for agentic AI workloads:

1. **Inference** — LLM API calls (latency, tokens, hosted vs local)
2. **Execution** — Tool calls (shell commands, file I/O, network)
3. **Orchestration** — Framework overhead (parsing, context management, routing)

## Usage

### 1. Run benchmark with sub-span instrumentation

The agent loop emits hierarchical sub-spans automatically when `run_context` is passed:

```python
from src.agent_loops.litellm_invoker import LiteLLMAgentInvoker
from src.benchmarks.terminal_bench import TerminalBenchAdapter
from src.runner import RunContext, track_span

ctx = RunContext(measurements=discover_measurements().values(), output_dir=output_dir)

with ctx:
    for spec in adapter.list_tasks(limit=10):
        with track_span(ctx, spec.id, kind="terminal_bench", node_id=spec.id):
            result = adapter.run_task(spec, agent_invoker=invoker, run_context=ctx)
```

### 2. Analyze results

```bash
poetry run agentsysperf analyze /tmp/agentsysperf_scratch/terminal_bench_phase_b
```

### 3. Read breakdown verdict

```
sample/count-lines
├── breakdown: inference_dominant (confidence: 0.90)
│   ├── Evidence: task_duration_s=5.64, inference_s=5.57, inference_pct=98.7, 
│   │   execution_s=0.03, execution_pct=0.5, orchestration_s=0.05, 
│   │   orchestration_pct=0.8, inference_calls=3, execution_calls=2
│   └── → Recommendations:
│       ├── • Inference latency dominates (99% of time)
│       ├── • Consider faster model (smaller or distilled) for lower latency
│       ├── • Explore batching or speculative decoding if multiple calls
│       └── • Profile network latency if using hosted LLM
├── cache: l3_pressure (confidence: 0.80)
...
```

## Interpretation

### Inference-Dominant (60-99% inference)

**Example:** Terminal-Bench with gpt-4o-mini (hosted)
- **Inference:** 98.7% (5.57s)
- **Execution:** 0.5% (0.03s)
- **Orchestration:** 0.8% (0.05s)

**Diagnosis:** Blocked on OpenAI API network latency  
**CPU classification:** `io_bound` (low CPU utilization, high IPC)  
**Optimization:**
- Faster model (gpt-3.5-turbo, claude-haiku)
- Local model deployment (vLLM, TGI)
- Request batching if multiple calls
- Async I/O for parallel requests

### Execution-Dominant (60-99% execution)

**Example:** Code compilation, large file operations
- **Inference:** 10% (tool planning)
- **Execution:** 85% (compile, test, build)
- **Orchestration:** 5%

**Diagnosis:** Slow tool calls (disk I/O, compilation, network fetch)  
**CPU classification:** `core_bound` or `io_bound` (depends on tool type)  
**Optimization:**
- Profile slow commands (`perf record`, `strace`)
- Async execution for parallelizable tasks
- Cache intermediate results (ccache for compilation)
- Use faster storage (NVMe vs SATA)

### Orchestration-Heavy (30-60% orchestration)

**Example:** Complex multi-turn loops with small tasks
- **Inference:** 30%
- **Execution:** 20%
- **Orchestration:** 50%

**Diagnosis:** Framework overhead (parsing, routing, state management)  
**Optimization:**
- Profile agent loop with `py-spy` or `cProfile`
- Reduce per-turn overhead (ReACT parsing, JSON validation)
- Optimize context window management
- Consider compiled orchestration (no Python interpreter per turn)

## Architecture

### Span Hierarchy

```
task_span: sample/count-lines (kind=terminal_bench)
├── turn_0_llm (kind=inference) ← litellm.completion()
├── turn_0_cmd (kind=execution) ← env.exec("ls -la")
├── turn_1_llm (kind=inference)
├── turn_1_cmd (kind=execution)
└── turn_2_llm (kind=inference)
```

Hierarchical span IDs: `{task_id}/turn_{N}_llm` or `{task_id}/turn_{N}_cmd`

### Measurement Capture

- **L1 (resource):** CPU%, RSS, threads, wall time (via `psutil` sampling)
- **L3 (perf counters):** IPC, cache miss%, branch miss%, LLC miss/s

Each sub-span gets L1+L3 measurements automatically via `track_span()`.

### Breakdown Calculation

1. **Task duration:** From task-level span L1 `duration_us`
2. **Inference time:** Sum of all sub-spans with `kind=inference`
3. **Execution time:** Sum of all sub-spans with `kind=execution`
4. **Orchestration time:** Residual = `task_duration - inference - execution`

Orchestration captures:
- ReACT parsing overhead
- Agent loop state management
- Environment provisioning (one-time setup)
- Conversation history management

## Example: 10-Task TB2 Run

```bash
export OPENAI_API_KEY=sk-...
poetry run python examples/run_terminal_bench_tb2.py --limit 10 --model gpt-4o-mini
poetry run agentsysperf analyze /tmp/agentsysperf_scratch/terminal_bench_tb2
```

**Expected results (gpt-4o-mini hosted):**
- Inference: 95-99% (network latency to OpenAI)
- Execution: 0.5-2% (shell commands fast on modern Xeon)
- Orchestration: 0.5-3% (LiteLLM + ReACT parser)

**CPU classification:** All tasks `io_bound` (low CPU utilization during inference wait)

**Cache behavior:** `l3_resident` (working set < 100 MB fits in L3)

## Implementation Details

### Agent Loop Instrumentation

`LiteLLMTerminalAgentLoop._run_turn()` emits sub-spans:

```python
# Wrap LLM call
span_id = f"{parent_span_id}/turn_{turn_idx}_llm"
with track_span(ctx, span_id, kind="inference", node_id=f"llm_call_{turn_idx}"):
    response = litellm.completion(...)

# Wrap command execution
cmd_span_id = f"{parent_span_id}/turn_{turn_idx}_cmd"
with track_span(ctx, cmd_span_id, kind="execution", node_id=f"cmd_{turn_idx}"):
    env_result = self._sync_env.exec(command, timeout_sec=...)
```

### BreakdownAnalyzer

`src/analyzers/breakdown.py`:
- Input: L1 measurement records (all spans)
- Output: Per-task breakdown verdict with evidence

Verdicts:
- `inference_dominant` (>60% inference)
- `inference_heavy` (40-60%)
- `execution_dominant` (>60% execution)
- `execution_heavy` (40-60%)
- `orchestration_heavy` (>30% orchestration)

## Limitations

1. **Orchestration as residual:** May include measurement gaps, not just framework overhead
2. **No token-level metrics yet:** LiteLLM response tokens not captured (future work)
3. **Single-threaded agent loops:** Parallel tool execution not instrumented
4. **Harbor environments:** Docker provisioning time counted in orchestration (one-time cost)

## Future Work

- Token count extraction from LiteLLM responses
- Per-tool breakdown (which commands are slowest?)
- Cost estimation (tokens × model pricing)
- Parallel execution instrumentation (async agent loops)
- Network telemetry (request size, response latency)
