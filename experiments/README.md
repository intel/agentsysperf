# experiments/

Standalone scaling and contention experiments. These go beyond the CLI's `agentsysperf sweep run` by varying NUMA placement, phase mixes, and collecting per-density EMON TMA.

## experiments/scaling/

Synthetic phase-mix agent workers with NUMA pinning + optional EMON system-wide contention measurement.

```bash
# Quick smoke test (2 densities, ~20s)
python -m experiments.scaling.run_experiment --quick

# Medium study (8 densities, ~3 min)
python -m experiments.scaling.run_experiment --medium

# Full matrix (all densities × placements × mixes, ~30 min)
python -m experiments.scaling.run_experiment --full
```

### What it measures

Each "agent" is a synthetic worker that executes a configurable phase mix (compile, io_heavy, balanced, mixed) pinned to a CPU set. The experiment:

1. Spawns N agents at each density level (1, 4, 8, 16, 32, 64, 96)
2. Pins each agent to a cpuset (intra_node, cross_node, or spread placement)
3. Measures per-agent throughput (turns/s) and mean turn latency
4. Optionally collects EMON TMA per density point

### Results

Output goes to `/tmp/agentsysperf_scaling/all_results.json` and renders in the dashboard under **Scaling → Synth Density Study** and **Scaling → HW Resources**.

### Key files

| File | Purpose |
|------|---------|
| `config.py` | Topology, density levels, NUMA placements |
| `pinning.py` | CPU affinity assignment |
| `agent_worker.py` | Per-agent workload (phase mix execution) |
| `orchestrator.py` | Spawn N agents + EMON collection |
| `analysis.py` | Scaling curves, inflection detection |
| `plotting.py` | Charts: throughput, IPC, LLC MPKI |
| `run_experiment.py` | CLI entry point (--quick / --medium / --full) |
