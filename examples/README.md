# examples/

End-to-end driver scripts demonstrating agentsysperf workflows. Each is standalone and can be run directly.

## Benchmarking

| Script | Purpose |
|--------|---------|
| `run_synthetic_l1_l3.py` | Synthetic CPU tasks with L1 + L3 perf-stat measurement |
| `run_synthetic_with_l3.py` | Synthetic CPU with per-PID L3 attribution |
| `run_tb2_simple.py` | Minimal Terminal-Bench run (requires Docker + API key) |
| `run_tb2_with_storage.py` | Terminal-Bench with full store persistence |
| `run_terminal_bench_phase_a/b/c.py` | Phased Terminal-Bench runs (admit → reason → act) |
| `run_terminal_bench_tb2.py` | Full Terminal-Bench 2 driver |

## Scaling & Sweeps

| Script | Purpose |
|--------|---------|
| `run_scaling_sweep_tb2.py` | Concurrency density sweep (--dry-run for synthetic, --fixture for real) |
| `run_six_phase_sweep.py` | Six-phase agent sweep with local model |
| `run_six_phase_smoke.py` | Quick smoke test for the six-phase agent |
| `plot_six_phase_sweep.py` | Plot results from six-phase sweep |

## EMON (Hardware Analysis)

EMON (Intel SEP-based TMA/EDP) collection is provided by the separate,
Intel-only `agentsysperf-emon` plugin, which ships its own example drivers.
It is not part of the vendor-neutral core.

## Phase Profiling

| Script | Purpose |
|--------|---------|
| `run_phase_profiler_tb2.py` | PhaseProfiler analysis (--dry-run available) |
| `validate_phase_profiler.py` | Validate phase profiler output |

## Reporting & Storage

| Script | Purpose |
|--------|---------|
| `demo_sqlite_store.py` | Demonstrate SQLite store API |
| `demo_pptx_report.py` | Generate a PowerPoint report |
| `demo_full_reporting_workflow.py` | Full end-to-end: run → analyze → report |
| `generate_report.py` | Report generation from existing run |
| `generate_full_report.py` | Comprehensive report with all sections |

## Validation

| Script | Purpose |
|--------|---------|
| `run_profile_verify.py` | Optimization profile engagement verification |
| `validate_calibration.py` | Validate analyzer threshold calibration |

## Data Files

| File | Purpose |
|------|---------|
| `tb2_workload_selection.txt` | Curated TB2 task list rationale |
| `execution_heavy_tb2_tasks.txt` | Tasks that stress execution (act phase) |
| `terminal_bench_task_sized_replay/` | Task-sized replay workflow + fixture |
