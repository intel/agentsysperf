# Analyzer Guide — AgentSysPerf 0.1.0

Analyzers are pure functions that derive insights from measurement records. This guide covers all 8 analyzers — the 7 in the vendor-neutral core plus the Intel-only `emon` plugin — with their input layers, use cases, and verdict interpretation.

## Quick Reference: Analyzer Selection

| Analyzer | Use Case | Input Layers | Scope | Output |
|----------|----------|--------------|-------|--------|
| **cpu_bound** | Core vs memory vs frontend bottleneck | `l1`, `l3` | Per-span | Verdict: `core_bound`, `memory_bound`, `frontend_starved`, `io_bound` |
| **cache** | Dominant cache tier limiting performance | `l3` | Per-span | Verdict: `l1_bound`, `l2_bound`, `l3_bound`, `dram_bound` |
| **memory_leak** | RSS growth detection | `l1` | Per-span | Verdict: `leak_detected`, `stable`, `growing` |
| **memory_bandwidth** | Memory subsystem bottleneck patterns | `l3`, `l1_system` | Per-span | Verdict: `weight_streaming`, `working_set_overflow`, `cross_numa_traffic`, `bandwidth_saturation`, `kv_cache_pressure`, `capacity_thrashing` |
| **breakdown** | Time/resource spent per span kind (phase) | `l1`, `l3` | Per-run aggregate | Breakdown table: wall-clock %, CPU %, IPC, cache miss per phase |
| **phase_profiler** | Per-pipeline-phase hardware characterization | `l1`, `l3` (phase-tagged) | Per-run aggregate | Phase trends + inflection point (when orchestration > inference) |
| **scaling** | Concurrency saturation knee + bottleneck | `l1_system` (across sweep points) | Multi-run sweep | Verdict: knee density + bottleneck (scheduler, CPU, memory, I/O) |
| **emon** _(Intel-only plugin)_ | Detailed TMA + workload classification | EMON CSV + `l1_system` | Per-run | 5-step pipeline: classification → triage → stall decomposition → solutions + evidence |

---

## Per-Span Analyzers

These analyzers fire automatically when you run `agentsysperf run` or `agentsysperf analyze`.

### 1. CPUBoundAnalyzer (`cpu_bound`)

**Purpose:** Classify the dominant CPU bottleneck using Top-Down Microarchitecture Analysis (TMA).

**Input layers:** `l1`, `l3`

**Required fields in L3 payload:**
- `ipc` — instructions per cycle
- `cache_miss_pct` — L3 miss percentage (0–100)
- `branch_miss_pct` — branch miss percentage (0–100)
- `llc_miss_per_s` — L3 misses per second

**Verdicts:**

| Verdict | Condition | Interpretation | Next Step |
|---------|-----------|-----------------|-----------|
| `memory_bound` | LLC misses > 10M/s **AND** cache miss > 20% | Memory subsystem is the bottleneck | See MemoryBandwidthAnalyzer for patterns; consider NUMA pinning, data tiling, or quantization |
| `core_bound` | CPU utilization ≥ 0.94 (cores) | CPU frontend/backend saturated | Profile with VTune or `perf c2c` for hotspots; consider workload specialization (AMX, AVX-512) |
| `frontend_starved` | Branch miss > 11% | Branch prediction or instruction cache exhausted | Reduce conditional branches; use profile-guided optimization (PGO) |
| `io_bound` | None of above | I/O or other external delay | Check `iowait_%` in `l1_system` records; tune system calls or I/O batching |

**Example:** A Language Model inference with IPC=0.8, cache_miss=35%, is classified as `memory_bound` (high miss, low throughput).

**Note on CPU utilization:** Measured in *cores* worth of CPU time per wall-clock second, not 0–1 normalized. A 4-thread workload on a 288-core host has utilization ~4.0, not 0.014.

---

### 2. CacheAnalyzer (`cache`)

**Purpose:** Determine which cache tier (L1, L2, L3, or DRAM) is the limiting factor.

**Input layers:** `l3`

**Required fields in L3 payload:**
- `l1_miss_per_s` — L1 misses per second
- `l2_miss_per_s` — L2 misses per second
- `llc_miss_per_s` — L3 misses per second
- `mem_read_per_s` — DRAM reads per second

**Verdicts:**

| Verdict | Condition | Interpretation |
|---------|-----------|-----------------|
| `l1_bound` | L1 miss rate > 5% of all memory operations | L1 cache too small or insufficient associativity |
| `l2_bound` | L2 miss rate high (20–50% of L1 misses hitting L2) | L2 is saturated; working set exceeds L2 budget |
| `l3_bound` | L3 miss rate high (50%+); LLC-misses >> DRAM traffic | L3 contention or poor cache locality |
| `dram_bound` | DRAM accesses dominate; LLC misses exceed local bandwidth | Main memory is the bottleneck; consider KV-cache compression or NUMA pinning |

**Example:** KV-cache pressure in LLM inference shows L3 miss rate climbing from 20% to 80% as context length grows → verdict: `dram_bound`. Solution: KV-cache compression or paged attention.

---

### 3. MemoryLeakAnalyzer (`memory_leak`)

**Purpose:** Detect memory leaks via RSS (resident set size) growth.

**Input layers:** `l1`

**Required fields in L1 payload:**
- `rss_mb` — resident set size, in MB
- Timeline: at least 3 snapshots (start, middle, end)

**Verdicts:**

| Verdict | Condition | Interpretation |
|---------|-----------|-----------------|
| `leak_detected` | RSS grows > 50 MB over span lifetime | Likely leak; object retention, buffer accumulation, or unclosed handles |
| `growing` | RSS grows 10–50 MB | Gradual growth; may be legitimate (buffer warming) or slow leak |
| `stable` | RSS change < 10 MB | No leak detected |

**Example:** A long-running agent orchestrator with 10 iterations shows RSS: [512 MB → 670 MB → 820 MB] → verdict: `leak_detected`, confidence 0.92.

**Caveat:** This is a heuristic. Legitimate workloads (buffer pre-allocation, JIT code gen) can grow. Always correlate with task count and iteration depth.

---

### 4. MemoryBandwidthAnalyzer (`memory_bandwidth`)

**Purpose:** Classify memory subsystem bottleneck patterns and map to solutions.

**Input layers:** `l3`, `l1_system`

**Required fields:**
- From L3: `ipc`, `cache_miss_pct`, working set estimate
- From L1 system: DRAM bandwidth utilization, NUMA hop counts

**Bottleneck Patterns:**

| Pattern | Signal | Example Workload | Solutions |
|---------|--------|------------------|-----------|
| **weight_streaming** | IPC < 1.5, cache miss > 70%, large RSS, sequential access | LLM forward pass (batch > L3 size) | Speculative decoding, INT8/INT4 quantization, model sharding, NUMA co-locality |
| **working_set_overflow** | Cache miss 40–70%, RSS > L3 budget, moderate IPC | Multi-agent orchestration with per-agent state | NUMA-aware scheduling, data tiling, L3 partitioning (Intel RDT/CAT) |
| **cross_numa_traffic** | Remote DRAM access > 20%, latency hop penalty visible | Agent threads scattered across NUMA nodes | NUMA pinning, memory-local scheduling, SNC mode tuning |
| **bandwidth_saturation** | DRAM BW > 70% of MLC peak, queuing delays rising | High-concurrency inference (agents/vCPU > 1) | Memory interleaving (STRIPE), workload spreading, DDR5 interleaving |
| **kv_cache_pressure** | RSS grows with context length, cache miss spikes mid-inference | Long-context retrieval-augmented generation (RAG) | KV-cache compression, paged attention, context pruning, RAG (reduce context) |
| **capacity_thrashing** | Cache miss rate oscillates, IPC unstable, LLC MPKI very high | Competing workloads with similar stride patterns | Cache partitioning (RDT/CAT), workload isolation, fewer co-tenants, CPU affinity |

**Example:** An agentic workload with 50+ concurrent agents shows DRAM BW=85%, IPC=0.6, cache miss=60% → verdict: `bandwidth_saturation` + `scheduler_oversubscription`. Solution: spread agents across DDR5 channels or reduce concurrency.

---

## Aggregate Analyzers

These produce run-level or sweep-level verdicts, not per-span ones.

### 5. BreakdownAnalyzer (`breakdown`)

**Purpose:** Decompose wall-clock time and hardware resources by span kind (phase).

**Input layers:** `l1`, `l3`

**Output:** A table of per-span-kind aggregates:

```
Span Kind           Wall-Clock %    CPU %    Avg IPC    Cache Miss %
────────────────────────────────────────────────────────────────────
retrieve            15.2%           12.1%    1.23       18%
reason              68.4%           71.3%    0.92       42%
act                 12.1%           10.2%    1.08       25%
commit              4.3%            6.4%     0.78       55%
```

**Use case:** Identify which pipeline phase consumes the most resources and has the worst hardware efficiency. Example: if "reason" (LLM inference) has IPC=0.92 but "retrieve" has IPC=1.23, optimize inference first.

---

### 6. PhaseProfiler (`phase_profiler`)

**Purpose:** Per-pipeline-phase hardware characterization and trend detection.

**Input layers:** `l1`, `l3` (requires phase-tagged spans)

**Phase tagging:** Spans must be wrapped with `phase="reason"`, `phase="act"`, etc.:

```python
with track_span("inference", phase="reason"):
    response = agent.invoke(query)
```

**Output:** Per-phase trends across iterations:

```json
{
  "phases": {
    "reason": {
      "wall_clock_pct": 68.4,
      "iterations": [23s, 24s, 23s, 46s, 49s],  // inflection at iter 3
      "avg_ipc": 0.92,
      "avg_cache_miss_pct": 42
    }
  },
  "inflection_points": [
    {
      "iteration": 3,
      "phase": "reason",
      "description": "Reason latency jumped 2x; act time also increased",
      "likely_cause": "context_accumulation"
    }
  ]
}
```

**Verdict interpretation:**

- **Latency inflection** (reason or act jumps): Context accumulation, task complexity spike, or cache eviction
- **Early reason spike** (iter 1–2): Model warm-up, JIT compilation
- **Gradual drift** (all phases grow): Memory leak, I/O backlog, or scheduler thrashing

**Use case:** Detect when an agentic workload switches from latency-bound to throughput-bound, or when garbage collection kicks in.

---

### 7. ScalingAnalyzer (`scaling`)

**Purpose:** Find concurrency saturation knee and classify the bottleneck at saturation.

**Input layers:** `l1_system` (aggregated across all densities in a sweep)

**Sweep execution:**

```bash
agentsysperf sweep run --dry-run   # writes sweep_points to SQLite
```

`--dry-run` cells are synthetic modeled points, so the knee it reports is a
property of the model, not of the box. Drop `--dry-run` and pass `--fixture` for
a measured sweep.

**Output:** One verdict per sweep:

```json
{
  "verdict": "cpu_bound",
  "confidence": 0.87,
  "evidence": {
    "knee": 1.5,
    "knee_density_desc": "1.5 agents per vCPU",
    "bottleneck": "cpu_bound",
    "bottleneck_reasoning": "cpu_peak 98%, runqueue stable, memory comfortable",
    "throughput_curve": "knee at 1.5x; flat beyond",
    "p95_latency_curve": "rises sharply after knee"
  }
}
```

**Bottleneck classifications:**

| Bottleneck | Signal | Mitigation |
|------------|--------|-----------|
| `scheduler_oversubscription` | runqueue_max >> logical_cpus | Reduce agent count; use pinning |
| `cpu_bound` | cpu_peak ~100%, cpu_avg high | Optimize compute: AMX, quantization, profile-guided opt |
| `memory_bound` | mem_avail_mb_min < 2 GB | Add DRAM; NUMA pinning; reduce per-agent state |
| `io_bound` | iowait_pct_avg > 15% | Batch I/O; async dispatch; tune network stack |
| `headroom_remaining` | None of above; knee not sharp | More headroom exists; profile for microoptimizations |

**Example:** A Terminal-Bench sweep from 0.25 to 3.0 agents/vCPU shows throughput peaking at 1.0, then flat. Verdict: `cpu_bound` at knee=1.0, confidence=0.92. **Next step:** Optimize the benchmark's hot loops with AMX or vectorization.

**Thresholds** (EMR estimates, adjust for your platform):
- `RUNQUEUE_OVERSUB_RATIO = 1.5` — runqueue > 1.5× cores
- `CPU_SATURATION_PEAK = 95%` — cpu_peak threshold
- `CPU_SATURATION_AVG = 80%` — cpu_avg threshold
- `MEM_LOW_MB = 2048` — low-memory threshold
- `IOWAIT_HIGH_PCT = 15%` — I/O-bound threshold

---

### 8. EmonAnalyzer (`emon`) — Intel-only plugin

> Provided by the separate, unreleased Intel-only `agentsysperf-emon` plugin,
> not by the vendor-neutral core. Install the plugin on an Intel host to
> register the `emon` measurement and analyzer; its own drivers collect the
> EMON CSV.

**Purpose:** 5-step EMON hardware counter analysis for detailed workload classification and TMA decomposition.

**Input layers:** EMON CSV output + `l1_system` telemetry

**Prerequisite:** the Intel-only `agentsysperf-emon` plugin installed, the SEP driver loaded, and its EMON collector run. EMON is not part of the vendor-neutral core.

**The 5-step pipeline:**

1. **Classification** — Workload type (vectorizable, memory-intensive, latency-sensitive, etc.)
2. **Triage** — TMA auto-detect from counters (Backend-Memory vs Frontend vs Core)
3. **Stall decomposition** — Which cycles are wasted: L1 miss, L2 miss, store forwarding, speculation
4. **Solution mapping** — Specific recommendations for the detected pattern
5. **Evidence collection** — Counter values supporting each step

**Output example:**

```json
{
  "verdict": "memory_bound",
  "confidence": 0.94,
  "classification": "vectorizable",
  "tma_bucket": "Backend-Memory",
  "evidence": {
    "llc_miss_ratio": 0.58,
    "dram_read_latency_cycles": 312,
    "vector_ratio": 0.42,
    "recommendation": "Quantize model weights (INT8); enable VNNI instructions"
  }
}
```

**Bottleneck classes (auto-detected from TMA):**

| Class | Counter Signal | Interpretation | Typical Solutions |
|-------|----------------|-----------------|------------------|
| **Core-bound** | Backend-Compute > 40% | Core pipeline saturation | Speculative decoding, INT8 quantization, profile-guided optimization |
| **Memory-bound** | Backend-Memory > 40% | Memory subsystem is limiting | NUMA pinning, KV-cache compression, paged attention |
| **Frontend-bound** | Frontend > 30% | Instruction fetch/decode bottleneck | Reduce branch misses; improve code locality; PGO |
| **Bad-Speculation** | speculation waste > 20% | Branch mispredicts/misses | Tune predictor; reduce conditional branches |

**When to use EMON:**

- You need fine-grained TMA (not just IPC + cache miss)
- You have a dedicated run with exclusive PMU access
- You want vendor-specific counter interpretation (e.g., E-core vs P-core on Hybrid architectures)
- You're tuning for a specific SKU (Emerald Rapids, Sierra Forest, etc.)

**Note:** EMON requires `sudo` and exclusive PMU access. Run it separately from regular benchmarks.

---

## Running Analyzers

### Automatic (on every run)

Analyzers that match input layers fire automatically:

```bash
agentsysperf run --benchmark synthetic_cpu --num-tasks 3
# Output includes analyzer verdicts inline
```

### Manual (offline analysis)

Re-analyze existing records without re-running:

```bash
agentsysperf analyze /path/to/measurements/  --format json
agentsysperf analyze /path/to/measurements/  --format tree
```

**Omitted from `agentsysperf analyze`:**
- `scaling` — requires sweep metadata (use `agentsysperf sweep run` instead)
- `emon` — provided by the Intel-only `agentsysperf-emon` plugin; requires a separate EMON CSV
- `phase_profiler` — requires phase-tagged spans; fires when present

---

## Verdict Structure

Every analyzer emits an `AnalysisResult`:

```python
@dataclass
class AnalysisResult:
    span_id: Optional[str]      # None for aggregate verdicts
    verdict: str                # e.g., "core_bound", "leak_detected"
    confidence: float           # 0.0–1.0; 0.5 = uncertain, 0.9+ = high confidence
    evidence: Dict[str, Any]    # Raw counters, thresholds, reasoning
    recommendations: List[str]  # Actionable next steps (e.g., ["Enable AMX", "Pin to NUMA node"])
```

**Example:**

```json
{
  "span_id": "span_1234",
  "verdict": "memory_bound",
  "confidence": 0.87,
  "evidence": {
    "llc_miss_per_s": 15_000_000,
    "cache_miss_pct": 58,
    "ipc": 0.72,
    "gate_1": "llc_miss_per_s (15M) > 10M ✓",
    "gate_2": "cache_miss_pct (58%) > 20% ✓"
  },
  "recommendations": [
    "Reduce model size or batch size (working set overflow)",
    "Enable NUMA pinning (affinity to local DRAM)",
    "Quantize to INT8 (reduce bandwidth)"
  ]
}
```

---

## Decision Trees: "What Should I Measure?"

### Goal: Optimize latency of a Terminal-Bench agent

1. **Measure L1 + L3** → run `agentsysperf run -b terminal-bench --num-tasks 2`
2. **Fire analyzers** → cpu_bound, cache, memory_leak, memory_bandwidth all emit
3. **Read verdicts**:
   - If `core_bound` → profile with VTune; optimize hot loop
   - If `memory_bound` → measure scaling; find knee; classify bottleneck at saturation
   - If `memory_leak` → check RSS growth; investigate object lifetime

### Goal: Find concurrency saturation

1. **Run scaling sweep** → `agentsysperf sweep run` (add `--dry-run` for a synthetic pipeline check)
2. **ScalingAnalyzer** → emits knee + bottleneck classification
3. **Next steps**:
   - Knee > 1.0 → CPU has headroom; add more agents or optimize other subsystem
   - Knee = 1.0 → saturation at 1 agent/vCPU; optimize within agent
   - Bottleneck = scheduler → reduce concurrency or pin threads

### Goal: Understand per-phase resource consumption

1. **Tag spans with phases** → `with track_span(..., phase="reason"):`
2. **Run benchmark** → `agentsysperf run -b terminal-bench --num-tasks 5`
3. **PhaseProfiler** → emits per-phase IPC, cache miss, wall-clock %
4. **Breakdown** → emits same table per span kind
5. **Correlate**: If "reason" has low IPC, optimize inference; if "retrieve" has high cache miss, optimize embeddings

---

## Configuring Analyzers

### Skipping an analyzer

Analyzers are discovered from installed packages and fire when their required
input layers are present. To stop one from firing, don't collect its input
layer (e.g. run without `perfspect`), or uninstall the package that provides it
(e.g. the optional Intel-only `agentsysperf-emon` plugin).

### Override thresholds

Thresholds are module-level constants (see `src/analyzers/<name>.py`). Edit locally for your platform:

```python
# src/analyzers/cpu_bound.py
CPU_SATURATION_PEAK = 95.0  # ← tune for your workload
```

Re-run analysis: thresholds take effect immediately.

### Plugin your own analyzer

See `docs/PLUGIN_DEVELOPMENT.md` → "Writing an Analyzer" for the entry-point template and registration.

---

## FAQ

**Q: Why did an analyzer emit no verdict?**

A: It likely couldn't find the required input layers. Check:
```bash
agentsysperf db show <run_id>  # see which layers were captured
```
If `l3` is missing, you need `kernel.perf_event_paranoid <= 1` or root access.

**Q: Why is my verdict confidence 0.5?**

A: The analyzer hit a boundary condition (e.g., IPC exactly at a threshold, cache miss exactly 20%). Rerun with slightly different workload parameters to disambiguate.

**Q: Can I run all analyzers at once?**

A: Yes. `agentsysperf run` and `agentsysperf analyze` auto-discover all registered analyzers. Disable specific ones with env vars or config if needed.

**Q: What if I don't have EMON / perf access?**

A: L1 analyzers (memory_leak, part of breakdown) still work. You'll miss L3-based verdicts (cpu_bound, cache, scaling). That's intentional: the suite honors "abort-don't-degrade" — better to emit fewer verdicts than incorrect ones.

**Q: How do I export verdicts for reporting?**

A: Use `agentsysperf report <run_id> --format md` or `--format pptx`. Verdicts are baked into the report with evidence and recommendations.

---

## References

- **Top-Down Microarchitecture Analysis (TMA):** https://www.intel.com/content/www/us/en/developer/articles/technical/top-down-microarchitecture-analysis.html
- **EMAT (Elastic Memorylatency Analyzer):** emat-benchmark on GitHub
- **Kneedle algorithm:** arXiv:1411.2312v1 (max-distance-from-chord knee detection)
- **AgentSysPerf Design:** `docs/methodology/DESIGN.md`
- **Integrity Principles:** README, *Integrity principles*
