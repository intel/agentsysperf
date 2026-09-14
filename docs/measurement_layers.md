# AgentSysPerf Measurement Layers

## Overview

AgentSysPerf uses a **5-layer measurement architecture** (L1-L5) that captures performance characteristics from application-level resource usage down to socket-level uncore counters. Each layer is implemented as a plugin conforming to the `Measurement` Protocol, enabling composable, independent measurement collection.

**Design principles:**
- **Composability:** Multiple layers run simultaneously on the same span without interference
- **Protocol-based:** All layers follow the `Measurement` Protocol (observe_span → finalize_span lifecycle)
- **Graceful degradation:** Missing hardware access or privileges downgrades to no-op, doesn't crash the run
- **Measured-not-assumed:** No fabricated counter values for unavailable events

---

## L1: Resource Metrics (✅ Implemented)

**What:** Process-level resource utilization (CPU%, RSS, threads, wall time)

**How:** psutil-based sampling at 20ms intervals, aggregated per span

**Capture:** Per-span statistics across the span's lifetime
- CPU time (user + system seconds consumed by span's thread)
- CPU% (mean and peak over sampled intervals)
- RSS (peak resident set size in KB)
- Thread count (peak concurrent threads)
- Sample count (number of psutil samples captured)

**Implementation:** `src/measurements/l1_subspan/probe.py`

**Entry point:** `l1_subspan = "src.measurements.l1_subspan:L1SubSpanMeasurement"`

**Example output:**
```python
MeasurementRecord(
    span_id="run-abc::task-1",
    layer="l1",
    payload={
        "kind": "synthetic_cpu",
        "node_id": "linalg",
        "duration_us": 2014332,
        "cpu_time_s": 1.98,
        "cpu_pct_mean": 99.1,
        "cpu_pct_peak": 100.0,
        "rss_kb_peak": 312448,
        "num_threads_peak": 4,
        "sample_count": 100,
    }
)
```

**Technical details:**
- Passive plugin: depends on `PsutilSampler` thread started by `RunContext`
- Thread-safe: sampler updates `SpanRecord` atomically via `SpanRegistry`
- No kernel privileges required
- Sub-5% overhead per independent harness validation

**Usage:**
```python
from src.measurements.l1_subspan import L1SubSpanMeasurement
from src.runner import RunContext

ctx = RunContext(measurements=[L1SubSpanMeasurement()], ...)
```

---

## L2: CPU Flame Graphs (❌ Not Started)

**What:** Call-stack profiling scoped to measurement spans

**How:** py-spy or perf-script flame graphs captured per span

**Planned capture:**
- Per-span flame graph (SVG or folded stack format)
- Top-N hot functions by CPU time
- GIL contention detection (Python-specific)

**Status:** Protocol examples mention L2 (protocols.py:322), no implementation yet

**Planned implementation:**
- `L2PySpy` plugin: wraps py-spy with span-scoped profiling
- Entry point: `l2_pyspy = "src.measurements.l2_pyspy:L2PySpyMeasurement"`
- Lifecycle: `observe_span` → start profiler, `finalize_span` → stop + emit SVG path

**Roadmap:** Not on critical path (Phase 1-3 focus on L1+L3 → analyzers → recommender)

**Why separate from L1:** L1 is always-on aggregate stats; L2 is expensive per-stack-frame recording, selectively enabled

---

## L3: Hardware Performance Counters (✅ Implemented)

**What:** CPU hardware counters (IPC, cache misses, branch mispredictions, context switches)

**How:** Linux `perf stat` with continuous interval sampling (100ms default), attributed to spans by wall-clock overlap

**Capture:** Per-span hardware counter aggregates
- Raw event counts (cycles, instructions, cache-references, cache-misses, branch-misses, etc.)
- Derived metrics (IPC, cache miss %, branch miss %, LLC misses/sec)
- Sample counts per event (for statistical confidence)

**Implementation:** `src/measurements/l3_perf/probe.py`

**Entry point:** `l3_perf = "src.measurements.l3_perf:L3PerfMeasurement"`

**Default events:**
```python
DEFAULT_EVENTS = [
    "cycles",
    "instructions",
    "cache-references",
    "cache-misses",
    "branch-instructions",
    "branch-misses",
]
```

**Example output:**
```python
MeasurementRecord(
    span_id="run-abc::task-1",
    layer="l3",
    payload={
        "kind": "synthetic_cpu",
        "node_id": "linalg",
        "duration_s": 2.014,
        "events": {
            "cycles": 4_800_000_000,
            "instructions": 3_400_000_000,
            "cache-references": 120_000_000,
            "cache-misses": 2_400_000,
        },
        "sample_counts": {"cycles": 21, "instructions": 21, ...},
        "ipc": 0.708,
        "cache_miss_pct": 2.0,
        "branch_miss_pct": 1.2,
        "llc_miss_per_s": 1_191_588,
    }
)
```

**Technical details:**
- Single `perf stat -I 100 -a` process runs for entire benchmark (low overhead)
- Interval samples written to CSV, parsed on span close
- Attribution by wall-clock overlap (span open/close times vs sample timestamps)
- Target modes: `system` (default), `self` (current PID), `pid:<N>`, `cgroup:<path>`
- Graceful degradation: if `perf` unavailable or `kernel.perf_event_paranoid` blocks access, plugin becomes no-op

**Usage:**
```python
from src.measurements.l3_perf import L3PerfMeasurement

# System-wide (default, diluted by other processes on large machines)
L3PerfMeasurement()

# Current process only (recommended for in-process benchmarks)
L3PerfMeasurement(target="self")

# Custom events + interval
L3PerfMeasurement(events=["cycles", "L1-dcache-loads", "L1-dcache-load-misses"], sample_interval_ms=50)
```

**Known limitations:**
- Generic events only (no vendor-specific AMX/TMUL counters yet — see L5)
- TMA (Top-down Microarchitecture Analysis) events require explicit opt-in (not in DEFAULT_EVENTS)
- Multi-socket attribution imprecise without per-socket cgroup isolation

---

## L4: Scheduler/SMT/GIL Diagnostics (❌ Not Started)

**What:** OS scheduler behavior, SMT contention, Python GIL load

**How:** Combination of `perf sched record`, `taskset` ablation, gil_load tracing

**Planned capture:**
- Context switches per span
- CPU migrations (cross-core, cross-socket)
- SMT sibling contention (thread 0 vs thread 1 on same core)
- GIL hold time distribution (Python-specific)
- Scheduler latency (wakeup latency p50/p99)

**Status:** Mentioned in protocols.py:325 ("L4 SMT/scheduler/GIL diagnostics"), no implementation

**Planned implementations:**
- `L4SchedPerf`: wraps `perf sched record` scoped to span windows
- `L4GILLoad`: eBPF-based GIL instrumentation (requires BCC/bpftrace)
- Entry points: `l4_sched = "src.measurements.l4_sched:L4SchedMeasurement"`

**Roadmap:** Phase 4 (Pattern Detectors) — after L1+L3+Analyzers ship

**Why separate from L1/L3:**
- L1 reports CPU% (aggregate); L4 explains *why* CPU% is low (scheduler starved, GIL contention)
- L3 reports IPC; L4 explains cross-core thrashing or hyperthreading interference

---

## L5: Socket/Uncore Counters (⏸️ Blocked)

**What:** Socket-level uncore metrics (memory bandwidth, RAPL energy, QPI/UPI interconnect)

**How:** Intel PCM (Performance Counter Monitor) for uncore PMU access + RAPL energy

**Planned capture:**
- Memory bandwidth (per-channel, per-socket, read/write breakdown)
- LLC occupancy (per-socket)
- QPI/UPI traffic (inter-socket data movement)
- RAPL energy (per-package, per-DRAM, per-core)
- AMX utilization (vendor-specific core counter, requires raw perf codes)
- oneDNN kernel dispatch ratio (vendor-specific)
- NUMA remote access ratio (cross-socket memory accesses)

**Status:** **BLOCKED on Intel PMU event codes for Xeon Platinum 8592+ (Emerald Rapids, model 207)**

**Blocker details:**
- PCM binary built at `~/Projects/pcm/build/bin/pcm` (works for uncore)
- Core-level vendor events (AMX_OPS_RETIRED, TMUL, oneDNN markers) need raw perf codes (cpu/event=0xXX,umask=0xYY/)
- Generic `perf list` doesn't expose vendor events for EMR
- Need vendor-provided PMU documentation or VTune SDK integration

**Partial workarounds available:**
- Hugepage faults: `/proc/vmstat` or `/proc/<pid>/smaps`
- NUMA stats: `/sys/devices/system/node/node*/numastat`
- QAT queue depth: `/sys/kernel/debug/qat` (if QAT drivers present)

**Roadmap:** After PMU codes obtained from the Intel perf team
- Option A: Extend `PerfStatTelemetry` with raw event map (cpuid model 207 → event codes)
- Option B: Create `intel_pcm` plugin combining PCM uncore API + raw perf codes
- Option C: Integrate VTune SDK (cleaner, Intel-supported API)

**Current HardwareTelemetry plugin:**
`PerfStatTelemetry` in `src/measurements/l3_perf/telemetry.py` exists for engagement verification (used by OptimizationProfilePlugin.verify_engaged), but only supports generic events.

**Entry point (when ready):** `intel_pcm = "src.measurements.intel_pcm:IntelPCMMeasurement"`

---

## Network Metrics (❌ Not Started)

**What:** Network-layer request metrics for distributed agent stacks (LLM API calls, gateway routing)

**Planned capture:**
- DNS resolution time
- TLS handshake time
- Time-to-first-byte (TTFB)
- Transfer time
- Connection reuse (pooled vs new)
- Retries, HTTP status codes
- Gateway latency, backpressure signals

**Status:** Planned as a separate `NetworkTelemetry` Protocol, not a `Measurement` layer. Not built — see "Unbuilt (roadmap, not shipped)" in the README.

**Why separate Protocol:**
- Measurement is span-scoped (time windows); network is request-scoped (start/end pairs may cross spans)
- Network failures propagate (connection refused → error); measurement failures degrade gracefully
- Needs correlation IDs for distributed tracing (X-B3-TraceId)

**Planned implementations:**
- `HTTPXTelemetry`: wraps httpx client with request hooks
- `OTELNetworkTelemetry`: reads from OpenTelemetry spans
- Entry point group: `agentsysperf.network_telemetry`

**Roadmap:** Phase 2 (3 weeks, parallel with Phase 1 Analyzer work)

**Current state:** No implementation. Terminal-Bench adapter integration will require this.

---

## Application-Level Metrics (❌ Not Implemented)

**What:** LLM-specific metrics (token counts, streaming latency, prompt/completion breakdown)

**Planned capture:**
- Prompt tokens, completion tokens
- Time-to-first-token (LLM streaming latency)
- Tokens per second (generation throughput)
- Model name, provider endpoint
- Tool invocations per turn

**Status:** Not captured at measurement layer yet. Could be:
1. Part of NetworkTelemetry (if treated as HTTP request metadata)
2. Part of BenchmarkAdapter.on_step callback (per-turn instrumentation)
3. New `L6ApplicationMetrics` layer (if standardized across adapters)

**Current approach:** Terminal-Bench adapter tracks turns/steps internally but doesn't emit as MeasurementRecords yet.

**Roadmap:** Phase 5 (Real Benchmark Adapters) will clarify where this belongs

---

## Sub-Span Instrumentation

**How hierarchical spans work:**

Adapters can open nested spans inside `BenchmarkAdapter.run_task()`:

```python
# Outer span: entire task (opened by runner)
with track_span(ctx, task.id, kind="task", node_id=task.id):
    # Inner span: LLM call
    with track_span(ctx, f"{task.id}::turn_1_llm", kind="llm", node_id="planner"):
        response = llm.invoke(prompt)
    # Inner span: command execution
    with track_span(ctx, f"{task.id}::turn_1_cmd", kind="bash", node_id="executor"):
        run_command(response.action)
```

**All measurement layers automatically observe each span:**
- L1 captures CPU%/RSS for outer task, LLM phase, cmd phase independently
- L3 captures IPC for each phase (answers "was IPC low during LLM wait or execution?")
- Analyzers can then detect patterns (e.g., "CPU idle during LLM phase, memory-bound during execution")

**Example span hierarchy (Terminal-Bench):**
```
task-1234                       (L1: 30s wall, 10s CPU)
  ├─ turn_1_llm                 (L1: 5s wall, 0.1s CPU; L3: low IPC — I/O bound)
  ├─ turn_1_cmd_bash            (L1: 2s wall, 1.8s CPU; L3: 1.2 IPC — core-bound)
  ├─ turn_2_llm                 (...)
  └─ turn_2_cmd_docker          (...)
```

**Implementation detail:** `track_span()` pushes/pops SpanRecords from a thread-keyed registry. Sampler and measurement plugins read the registry's active spans without needing adapter-specific wiring.

---

## Measurement Lifecycle

All layers follow the same Protocol:

1. **start(run_id, output_dir)** — Initialize for the run
   - L1: Start psutil sampler thread
   - L3: Spawn `perf stat -I 100` subprocess
   - L5: Initialize PCM (when implemented)

2. **observe_span(span_id, kind, node_id)** — Span opened
   - L1: Push SpanRecord to registry
   - L3: Record span open timestamp
   - L5: Mark span boundary for uncore attribution

3. **finalize_span(span_id) → Iterable[MeasurementRecord]** — Span closed
   - L1: Pop SpanRecord, emit aggregate stats
   - L3: Slice interval CSV, sum counters, compute IPC
   - L5: Read PCM snapshot for span window

4. **stop() → Iterable[MeasurementRecord]** — End of run
   - L1: Stop sampler thread
   - L3: SIGINT perf subprocess, flush CSV
   - L5: Cleanup PCM

**Error handling:** Plugin exceptions are caught and logged by `RunContext`. A buggy L3 plugin never crashes L1 or the benchmark.

---

## Extensibility

### How to add a new measurement layer

1. **Implement the Protocol** (`src/protocols.py:Measurement`):
   ```python
   from src.protocols import Measurement, MeasurementRecord
   
   class L6MyMeasurement:
       name: str = "my_layer"
       layer: str = "l6"
       
       def start(self, *, run_id: str, output_dir: Path) -> None:
           # Initialize resources
       
       def observe_span(self, *, span_id: str, kind: str, node_id: str) -> None:
           # Optional: track span open
       
       def finalize_span(self, span_id: str) -> Iterable[MeasurementRecord]:
           # Emit records for this span
           yield MeasurementRecord(span_id=span_id, layer=self.layer, payload={...})
       
       def stop(self) -> Iterable[MeasurementRecord]:
           # Cleanup, emit any final records
           return ()
   ```

2. **Register entry point** in `pyproject.toml`:
   ```toml
   [project.entry-points."agentsysperf.measurements"]
   my_layer = "my_package.my_measurement:L6MyMeasurement"
   ```

3. **Discovery:** `agentsysperf list` will auto-discover the plugin via entry points

4. **Use in benchmarks:**
   ```python
   from my_package.my_measurement import L6MyMeasurement
   from src.runner import RunContext
   
   ctx = RunContext(measurements=[L1SubSpanMeasurement(), L6MyMeasurement()], ...)
   ```

**Reference implementation:** Read `src/measurements/l1_subspan/probe.py` as the simplest complete example.

---

## Entry Point Registration

Current registered measurements (from `pyproject.toml`):

```toml
[project.entry-points."agentsysperf.measurements"]
l1_subspan = "src.measurements.l1_subspan:L1SubSpanMeasurement"
l3_perf = "src.measurements.l3_perf:L3PerfMeasurement"
```

Discovery API:
```python
from src.protocols import discover_measurements

all_measurements = discover_measurements()  # Returns dict[str, Measurement]
print(all_measurements.keys())  # ['l1_subspan', 'l3_perf']
```

CLI:
```bash
$ poetry run agentsysperf list
Registered measurements: l1_subspan, l3_perf
```

---

## Example: Composing L1 + L3

From `examples/run_synthetic_l1_l3.py`:

```python
from src.measurements.l1_subspan import L1SubSpanMeasurement
from src.measurements.l3_perf import L3PerfMeasurement
from src.runner import RunContext, track_span

ctx = RunContext(
    measurements=[
        L1SubSpanMeasurement(),
        L3PerfMeasurement(target="self"),  # Attach to current process
    ],
    output_dir=Path("/tmp/agentsysperf_scratch/synthetic_l1_l3"),
)

with ctx:
    for task in adapter.list_tasks():
        with track_span(ctx, task.id, kind="synthetic_cpu", node_id=task.id):
            adapter.run_task(task, agent_invoker=invoker)

# ctx.records now contains both L1 and L3 MeasurementRecords
l1_records = [r for r in ctx.records if r.layer == "l1"]
l3_records = [r for r in ctx.records if r.layer == "l3"]

for r in l1_records:
    print(f"Task {r.span_id}: CPU {r.payload['cpu_pct_mean']:.1f}%, RSS {r.payload['rss_kb_peak']/1024:.0f}MB")

for r in l3_records:
    print(f"Task {r.span_id}: IPC {r.payload.get('ipc', 'N/A')}, Cache miss {r.payload.get('cache_miss_pct', 'N/A'):.2f}%")
```

**Output:**
```
Task run-abc::linalg: CPU 99.1%, RSS 305MB
Task run-abc::linalg: IPC 0.71, Cache miss 2.00%
Task run-abc::compile: CPU 87.3%, RSS 1204MB
Task run-abc::compile: IPC 1.24, Cache miss 5.32%
```

**Zero cross-layer wiring required.** Each layer emits independently; analyzers compose them downstream.

---

## Storage and Analysis

**Serialization:** On `RunContext.stop()`, all MeasurementRecords are serialized to `{output_dir}/measurement_records.json`:

```json
[
  {
    "span_id": "run-abc::task-1",
    "layer": "l1",
    "payload": {
      "duration_us": 2014332,
      "cpu_time_s": 1.98,
      "cpu_pct_mean": 99.1,
      ...
    }
  },
  {
    "span_id": "run-abc::task-1",
    "layer": "l3",
    "payload": {
      "ipc": 0.708,
      "cache_miss_pct": 2.0,
      ...
    }
  }
]
```

**Analysis:** Analyzers consume these records (see `src/analyzers/`):

```python
from src.analyzers import CPUBoundAnalyzer, CacheAnalyzer

cpu_analyzer = CPUBoundAnalyzer()
cache_analyzer = CacheAnalyzer()

# Load records from JSON
with open(output_dir / "measurement_records.json") as f:
    records = [MeasurementRecord(**r) for r in json.load(f)]

# Derive insights
cpu_results = list(cpu_analyzer.analyze(records))
cache_results = list(cache_analyzer.analyze(records))

# Example: "Task 'linalg' is memory-bound (IPC=0.71), L3-resident (98% hit rate)"
```

---

## Known Limitations

1. **L3 multi-socket attribution:** System-wide perf dilutes counts by all cores; use `target="self"` or cgroups for accurate per-task counts on large machines

2. **L5 vendor events blocked:** No AMX/TMUL/oneDNN counters until Intel PMU codes available

3. **Network metrics missing:** Phase 2 work (NetworkTelemetry Protocol, not a Measurement layer)

4. **TMA counters not default:** Top-down Microarchitecture Analysis events (frontend-bound, backend-bound) require explicit opt-in via custom event list

5. **eBPF/BCC requirement for L4 GIL:** Requires kernel >=4.4, bpftrace installed (not available in all container environments)

---

## Next Steps

**Phase 1 (Week 1-4, ⭐ CRITICAL PATH):** Analyzer Protocol + Core Analyzers
- CPUBoundAnalyzer: L1+L3 → "core-bound" | "memory-bound" verdict
- CacheAnalyzer: L3 → dominant cache tier
- MemoryLeakAnalyzer: L1 RSS growth → leak detection

**Phase 2 (Week 2-5, parallel):** NetworkTelemetry Protocol
- HTTPXTelemetry reference implementation
- Terminal-Bench integration

**Phase 3 (Week 5-9, ⭐ CRITICAL PATH):** Recommender Protocol + Xeon SKU database
- Map analysis insights → SKU recommendations

**Phase 5 (Week 10-15, ⭐ CRITICAL PATH):** Real benchmark adapters
- Terminal-Bench, SWE-Bench integration

**L5 unblocked when:** Intel PMU event codes obtained (reach out to the Intel perf team)

---

**Last updated:** 2026-05-28  
**Status:** L1+L3 implemented and validated, L2/L4/L5 planned, Network metrics pending Phase 2
