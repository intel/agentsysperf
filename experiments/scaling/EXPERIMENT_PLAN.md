# Parallel Agent Density Scaling Experiment

## Goal

Measure how per-agent performance degrades as we pack more concurrent agentic workloads onto the same Granite Rapids Xeon socket, with EMON capturing the system-wide contention picture at each density level.

## Hardware

- **CPU**: Intel Xeon Granite Rapids (model 173), Redwood Cove cores
- **Cores**: 96 physical / 192 threads, single socket
- **NUMA (SNC3)**: node0=cores 0-31, node1=cores 32-63, node2=cores 64-95
- **Cache**: 480MB L3 (shared), 2MB L2/core, 48KB L1D/core
- **Turbo**: 4.2 GHz all-core
- **Memory**: DDR5-6400

## Experiment Matrix

| Dimension | Values | Rationale |
|-----------|--------|-----------|
| Agent density (N) | 1, 2, 4, 8, 12, 16, 24, 32 | Log-scale + inflection probes up to filling all 3 NUMA nodes |
| Phase mix | `compute_heavy`, `io_heavy`, `balanced`, `mixed` | Different agent workload profiles |
| NUMA placement | `intra_node`, `cross_node`, `spread` | Tests L3 sharing vs NUMA penalty |
| Cores per agent | 4 (default), 8, 12 | Working set vs cache partition tradeoff |

**Total configurations**: 8 densities x 4 mixes x 3 placements = 96, plus 4 solo baselines.

## Workload Mixes

### A. `compute_heavy` (90% Reason / 10% Act)

Simulates heavy inference (transformer weight streaming).

- **Reason phase**: 8MB array traversal, 16 passes per turn, stride-64 access pattern
- **Act phase**: trivial `echo` commands
- **Expected signature**: Backend_Bound/Memory_Bound, high IPC when un-contended, degrades to MEM_Bandwidth under L3 pressure

### B. `io_heavy` (20% Reason / 80% Act)

Simulates shell-heavy agentic execution (code generation + testing loops).

- **Reason phase**: 64KB array traversal (fits L1)
- **Act phase**: real file I/O — `find`, `grep`, `sort` over a generated 10MB directory tree per agent
- **Expected signature**: low CPU utilization, high context-switch rate, I/O-wait dominant

### C. `balanced` (60% Reason / 40% Act)

Realistic agentic pattern (inference + tool use).

- **Reason phase**: 2MB array traversal (fills L2, spills to L3)
- **Act phase**: mix of compute (`wc -l`, `awk`) and I/O (`find`, `du`) commands
- **Expected signature**: mixed TMA, moderate L3 pressure

### D. `mixed` (heterogeneous fleet)

Different agents run different mixes simultaneously.

- Agents 0..N/3: compute_heavy
- Agents N/3..2N/3: io_heavy
- Agents 2N/3..N: balanced
- **Tests**: interference between dissimilar workloads sharing L3

## Core Pinning Strategy

### NUMA Topology (SNC3)

```
node0: cores 0-31   (physical), threads 0-31 + 96-127 (HT)   → 160MB L3 slice
node1: cores 32-63  (physical), threads 32-63 + 128-159 (HT) → 160MB L3 slice
node2: cores 64-95  (physical), threads 64-95 + 160-191 (HT) → 160MB L3 slice
```

### Placement Strategies

1. **`intra_node`**: All N agents packed onto node0 only (cores 0-31). Tests maximum L3 contention within a single 160MB L3 slice. At N=8 with 4 cores/agent = 32 cores, fully saturating node0.

2. **`cross_node`**: Agents spread evenly across node0 and node1. Tests NUMA boundary effects when agents share no L3 slice but may share DRAM channels.

3. **`spread`**: Agents distributed round-robin across all 3 nodes. Least L3 contention but most NUMA traffic if agents access shared state.

### Reserved Cores

```
Orchestrator + EMON overhead: cores 92-95 (4 cores on node2)
Agent pool: cores 0-91 (92 cores available)
```

## Measurement Collection Plan

### Layer 1: EMON (system-wide, one per config)

- Full EDP collection during workload execution
- Post-process via pyEDP → 516+ derived metrics
- Key metrics: complete TMA tree, memory BW (MB/s), LLC demand miss latency (ns), NUMA local/remote %, package power (W), operating frequency (GHz)
- **Timing**: start 2s before workload, stop 1s after → captures warmup + cooldown

### Layer 2: L3 perf counters (per-agent PID)

- Attached to each agent process via `perf stat -p <pid>`
- Events (18): cycles, instructions, branch-instructions, branch-misses, cache-references, cache-misses, LLC-loads, LLC-load-misses, L1-dcache-loads, L1-dcache-load-misses, dTLB-load-misses, iTLB-load-misses, node-loads, node-load-misses, LLC-stores, LLC-store-misses, context-switches, page-faults
- Sample interval: 10ms
- Per-span records via `track_span`

### Layer 3: L1 timing (per-agent)

- L1SubSpanMeasurement: wall-clock, CPU time, RSS peak, thread count per span
- Enables PhaseProfiler Reason/Act breakdown per agent

### Layer 4: System telemetry (orchestrator-side)

- `vmstat 1` for context-switch rate, run queue depth
- `/proc/meminfo` snapshots for NUMA page distribution
- `numastat -p <pid>` per agent for NUMA page fault attribution

## Execution Protocol

```
1. Pre-run:    Start EMON EDP collection (2s warmup, no workload)
2. Spawn:      Create N agent worker processes, pin each to assigned cpuset
3. Barrier:    All agents blocked on multiprocessing.Barrier(N+1)
4. Start:      Orchestrator releases barrier → all agents begin simultaneously
5. Execute:    Each agent runs 8-turn TB2 loop (variable duration under contention)
6. Complete:   All agents finish (timeout: 60s for stragglers)
7. Cooldown:   1s additional EMON collection
8. Stop:       Stop EMON, collect all per-agent results via multiprocessing.Queue
9. Process:    pyEDP → metrics CSV, aggregate per-agent records
```

## Key Questions to Answer

1. **Scaling curve**: How does per-agent throughput (turns/sec) degrade vs solo baseline?
2. **L3 inflection**: At what density does LLC MPKI jump >2x?
3. **TMA shift**: Does Memory_Bound grow from ~20% (solo) to >50% (saturated)?
4. **NUMA penalty**: Does `intra_node` hit the wall earlier, or does `cross_node`'s latency tax dominate?
5. **Phase interaction**: Does compute_heavy suffer more from density than io_heavy?
6. **Optimal density**: What's the sweet spot (max aggregate throughput before per-agent latency becomes unacceptable)?

## Analysis Pipeline

### Step 1: Raw Data (per config)

- EMON CSV (system-wide, 516 metrics)
- Per-agent JSON records (L1 + L3 per span)
- vmstat/numastat time-series

### Step 2: Per-Agent Metrics Derivation

- IPC, LLC MPKI, L1D miss %, NUMA remote %, throughput (turns/s)
- PhaseProfiler breakdown (Reason % vs Act % vs overhead)
- Agent completion time distribution (mean, p50, p95, max)

### Step 3: Scaling Curve Generation

- X-axis: density (N)
- Y-axes: per-agent throughput normalized to solo baseline
- Grouped by phase_mix and placement strategy

### Step 4: Contention Inflection Detection

- Identify density at which LLC MPKI jumps >2x vs solo baseline
- Correlate with TMA Backend_Bound/Memory_Bound crossing threshold
- Per-placement: does intra_node hit the wall at lower density than spread?

### Step 5: NUMA Effect Quantification

- Compare intra_node vs cross_node vs spread at same density
- NUMA penalty coefficient: (cross_node_latency - intra_node_latency) / intra_node_latency
- Correlate node-load-misses with throughput degradation

### Step 6: TMA Heatmap

- Rows: density levels (1..32)
- Columns: TMA categories
- Cell color: % pipeline slots
- One heatmap per (mix, placement) combination

## Output Deliverables

### A. Scaling Curves (4 plots)

1. **Throughput vs Density** — per-agent normalized throughput (turns/s / solo_turns/s) grouped by mix
2. **IPC vs Density** — per-agent mean IPC grouped by placement
3. **LLC MPKI vs Density** — log-scale, contention inflection annotated
4. **Completion time vs Density** — p50 + p95 bands

### B. TMA Heatmaps (12 heatmaps)

- 4 mixes x 3 placements
- Shows TMA shift with density
- Key insight: does Backend_Bound/Memory_Bound grow from ~20% (solo) to >50% (saturated)?

### C. NUMA Comparison Bar Chart

- For each density, 3 bars (intra/cross/spread) showing throughput
- Annotated with NUMA remote access %

### D. Phase Breakdown Stacked Area

- X-axis: density
- Y-axis: % of wall-clock
- Stacked areas: Reason, Act, Overhead (barrier wait, OS scheduling)
- Shows whether Act phase degrades faster than Reason under contention

### E. Summary JSON

- Full experiment matrix results
- Inflection points identified
- Ranked bottlenecks per configuration
- Compatible with SQLite result store

## Implementation File Structure

```
experiments/scaling/
    EXPERIMENT_PLAN.md        # This file
    __init__.py
    config.py                 # ExperimentConfig, AgentConfig, PlacementStrategy enum, NUMA constants
    workloads.py              # 4 workload mix implementations
    agent_worker.py           # Single-agent subprocess entry point (pin → barrier → TB2 → return)
    pinning.py                # CPU affinity assignment for 3 placement strategies
    orchestrator.py           # Spawns agents, manages EMON, barriers, collection
    analysis.py               # Post-processing: aggregate, derive metrics, detect inflection
    plotting.py               # Generate all charts (matplotlib/seaborn)
    run_experiment.py         # CLI entry point: parse args, iterate matrix, call orchestrator
```

## Implementation Sequence

### Phase 1: Foundation

1. `config.py` — dataclasses, NUMA topology, density levels
2. `pinning.py` — `assign_cpuset()` returning `set[int]` per agent

### Phase 2: Single-Agent Worker

3. `workloads.py` — four workload classes with configurable array sizes and shell commands
4. `agent_worker.py` — subprocess entry point with self-pinning, RunContext, results via Queue

### Phase 3: Orchestrator

5. `orchestrator.py` — spawn N workers, EMON start/stop, barrier management, result collection

### Phase 4: Analysis + Visualization

6. `analysis.py` — aggregate results, compute scaling curves, detect inflection points
7. `plotting.py` — generate all charts

### Phase 5: Integration

8. `run_experiment.py` — CLI that iterates full matrix, invokes orchestrator, generates report

## Design Decisions

1. **`multiprocessing.Process` (not threading)**: Each agent must be a separate OS process for independent CPU affinity and per-PID perf counter attachment.

2. **Shared-nothing isolation**: Each agent gets its own `StandaloneEnvironment` (temp dir), `RunContext`, and `L3PerfMeasurement`. No shared state except the barrier.

3. **EMON is system-wide**: Cannot do per-agent EMON. Per-agent granularity comes from L3PerfMeasurement per-PID. EMON gives the "big picture" TMA including interference.

4. **Barrier-synchronized start**: All agents must begin simultaneously for EMON to capture the true concurrent-load pattern.

5. **Graceful degradation**: If EMON is unavailable, experiment still runs with L3 perf per-agent only.
