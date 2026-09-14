# docs/

Design documentation, methodology references, and guides.

## Guides (start here)

| Document | Purpose |
|----------|---------|
| `GETTING_STARTED.md` | Day-1-to-4 onboarding: install, run, understand, optimize |
| `ANALYZER_GUIDE.md` | All 7 core analyzers (+ the Intel-only emon plugin): use cases, verdicts, decision trees |
| `PLUGIN_DEVELOPMENT.md` | How to write benchmarks, measurements, and analyzers |

## Methodology

| Document | Purpose |
|----------|---------|
| `methodology/DESIGN.md` | 16-section design rationale |
| `methodology/hw_metrics_catalog.md` | Hardware metrics catalog (what each counter means) |
| `methodology/platform_capability_policy.md` | Platform detection and capability gating policy |
| `benchmark/methodology.md` | Replay benchmark methodology (fixture recording, proxy canonicalization) |

## Architecture

| Document | Purpose |
|----------|---------|
| `STORAGE_LAYER.md` | SQLite schema, migrations, DuckDB accelerator |
| `measurement_layers.md` | L1/L3/L5 measurement layer design |
| `adr/` | Architecture Decision Records |
| `contracts/` | Original protocol references, routing strategies, study YAMLs |

## Operational

| Document | Purpose |
|----------|---------|
| `FIXTURES.md` | Fixture registry (which fixtures exist, their provenance) |
| `RUNNING_FIXTURES.md` | How to deploy and run the replay proxy on remote hosts |
