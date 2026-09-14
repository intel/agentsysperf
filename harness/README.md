# AgentSysPerf Local Harness

Characterize agentic workflow stages by hardware signature on local Xeon silicon.

## What this does

Runs Terminal-Bench 2 tasks through Harbor, instruments every stage with
perf counters (cgroup-filtered per container) and system monitors, then
classifies each task by its hardware behavior. Produces per-task fingerprints
that answer: **what does each part of an agentic workflow actually need from
the CPU, and what optimizations (NUMA, hugepages, core isolation) help which
task types?**

## Prerequisites

- Linux x86_64 with `perf_event_paranoid <= 1` (for hardware counter access)
- Docker (for Harbor/Terminal-Bench task containers)
- Run `agentsysperf preflight` to verify system readiness

## Current Status

```
[x] Replay proxy (off / replay / record modes) — tested, working
[x] Perf sampler module (perf_event_paranoid=-1) — working
[x] Optimization profile applier (numactl, taskset) — working
[x] Stage inference module — working
[x] Runner framework — working
[x] Harbor 0.7.1 — installed in the repo venv at `.venv/bin/harbor`
[x] Terminal-Bench 2 — 89 tasks downloaded
[x] Synthetic workloads — 9 tasks, baselined with perf
[x] Oracle agent characterization — 5 tasks profiled with cgroup perf
[x] NUMA comparison — measured across all tasks (synthetic + real TB2 on nodes 0,1,2)
[x] Contention analysis — multi-task co-scheduling interference measured (real TB2)
[x] Multi-agent density test — 4/8/16 concurrent agents, contention cliff at n=16
[x] Multi-task contention — 3 task pairs co-scheduled, validated routing policy
[x] SLM interleaving — Qwen2.5-1.5B BF16/AMX, memory-BW bottleneck identified
[x] Cross-node SLM verification — NUMA isolation eliminates contention (~17-18 tok/s)
[x] Routing policy specification — consolidated all findings into actionable spec
```

## Expected Output

After running, each task gets classified by hardware character (IPC, cache
behavior, branch miss rate). Run `agentsysperf run --benchmark synthetic_cpu`
to generate your own baseline, or see the dashboard for interactive results.

## Setup

```bash
# 1. Unlock perf counters (one-time, persists until reboot)
sudo sysctl kernel.perf_event_paranoid=-1

# 2. Harbor (installed from source at ~/harbor-src, venv at ~/harbor-venv)
source ~/harbor-venv/bin/activate
harbor --version   # 0.7.1

# 3. Terminal-Bench 2 dataset (already downloaded)
ls datasets/terminal-bench-2/   # 89 tasks

# 4. Python venv for harness scripts
source .venv/bin/activate
```

## Run

```bash
cd <repo>/harness
source .venv/bin/activate

# --- Synthetic workloads (no Docker, fast validation) ---
python scripts/synthetic_tasks.py                  # all 9 tasks, ~30s
python scripts/synthetic_tasks.py --task compile   # single task
python scripts/synthetic_tasks.py --perf           # with perf stat per task

# --- Real TB2 tasks via Harbor (Docker required) ---
# Single task with oracle agent (no LLM needed)
~/harbor-venv/bin/harbor run \
  -p datasets/terminal-bench-2/train-fasttext \
  -a oracle -n 1 --yes

# With perf cgroup monitoring (attach to container)
# See scripts in results/oracle_final/ for the full workflow

# --- Runner framework (orchestrates proxy + monitors + harbor) ---
python scripts/runner.py --llm-mode off --tasks 3
python scripts/runner.py --llm-mode replay --profile numa_pinned

# --- Stage inference on completed results ---
python scripts/stage_inference.py results/base_off_c5/
```

## LLM Modes

| Mode | What happens | Use for |
|------|-------------|---------|
| `off` | Proxy returns empty instantly | Characterize orchestration + tool execution only |
| `replay` | Serve from recorded fixture | Deterministic full-trajectory replay |
| `slm` | Local small model on-CPU | Characterize on-CPU inference (future) |
| `live` | Forward to real endpoint | End-to-end measurement (future) |

## Output

```
results/<profile>_<mode>_c<N>/
├── stats.txt                  # summary: elapsed, trials, reward
├── task_phases.json           # per-trial timing from Harbor
├── stage_classification.json  # per-turn hardware class + signature
├── stage_summary.json         # aggregate per-class statistics
├── harbor_stdout.txt
├── harbor_stderr.txt
├── jobs/                      # Harbor output (result.json per trial)
└── monitoring/
    ├── mpstat.txt             # 1s CPU samples
    ├── vmstat.txt             # 1s memory/context-switch samples
    ├── docker_stats.txt       # per-container CPU%/mem
    └── perf_continuous.csv    # 100ms perf counter samples (if available)
```

## Profiles (optimization sweep)

| Profile | NUMA | Hugepages | Core isolation | What it tests |
|---------|------|-----------|----------------|---------------|
| `base` | unpinned | 4k | no | Stock default — the baseline |
| `numa_pinned` | node0 | 4k | no | NUMA-local allocation benefit |
| `numa_hugepages` | node0 | 2M | no | + TLB pressure reduction |
| `isolated` | node0 | 2M | cores 0-21 | + scheduling noise elimination |

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         AgentSysPerf Local Harness                        │
│                                                                        │
│  Goal: Characterize agentic workloads by hardware signature.          │
│  Method: Run tasks → collect perf counters → classify → compare       │
│          across optimization profiles.                                 │
└──────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────┐
                    │       runner.py          │
                    │  orchestrates full flow  │
                    └────────┬────────────────┘
                             │
              ┌──────────────┼──────────────────┐
              │              │                  │
              ▼              ▼                  ▼
┌─────────────────┐  ┌──────────────┐  ┌────────────────────┐
│  replay_proxy   │  │  Harbor 0.7  │  │  synthetic_tasks   │
│  (off/replay/   │  │  + Oracle/   │  │  (no Docker,       │
│   record)       │  │  Terminus-2  │  │   direct perf)     │
│                 │  │              │  │                    │
│  Serves LLM    │  │  Runs TB2    │  │  9 workloads:      │
│  responses or  │◄─│  tasks in    │  │  compile, ml_train │
│  returns empty  │  │  Docker      │  │  linalg, io,       │
└─────────────────┘  └──────┬───────┘  │  compress, ray,    │
                            │          │  sat, interp, ctrl │
                            │          └────────────────────┘
                            ▼
              ┌──────────────────────────────────┐
              │      monitoring layer             │
              │                                  │
              │  perf stat --cgroup (per-container, hardware counters)
              │  mpstat (1s system-wide CPU)     │
              │  vmstat (1s memory/ctx-switch)   │
              │  docker stats (per-container)    │
              └──────────────┬───────────────────┘
                             │
                             ▼
              ┌──────────────────────────────────┐
              │      analysis layer              │
              │                                  │
              │  stage_inference.py              │
              │    → classify: compute_heavy /   │
              │      memory_bound / io_bound /   │
              │      idle_wait / control_plane   │
              │                                  │
              │  optimization.py                 │
              │    → profiles: base /            │
              │      numa_pinned / hugepages /   │
              │      isolated                    │
              │                                  │
              │  NUMA comparison                 │
              │  Contention analysis             │
              └──────────────────────────────────┘
                             │
                             ▼
              ┌──────────────────────────────────┐
              │        results/                   │
              │                                  │
              │  synthetic_baseline/             │
              │    fingerprints.json             │
              │    CHARACTERIZATION.md           │
              │    CONTENTION_ANALYSIS.md        │
              │                                  │
              │  oracle_final/                   │
              │    fingerprints_final.json       │
              │    TB2_CHARACTERIZATION.md       │
              │    <task>/perf_cgroup.txt        │
              │                                  │
              │  ROUTING_ANALYSIS.md             │
              └──────────────────────────────────┘
```

## What This Measures

1. **Synthetic baseline** — 9 workloads approximating TB2 task CPU signatures,
   run directly under perf stat. Establishes per-task hardware fingerprints
   (IPC, cache miss rate, branch miss rate) without Docker overhead.

2. **NUMA comparison** — Synthetic tasks run across NUMA placements to
   quantify locality benefits per workload type.

3. **Contention analysis** — Pairs of tasks run concurrently to measure
   cache interference and identify which co-scheduling combinations degrade.

4. **Real TB2 characterization** — Tasks run through Harbor with perf stat
   attached to Docker container cgroups for per-container hardware counters.

Run the experiments on your own hardware to generate platform-specific results.
