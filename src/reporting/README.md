# AgentSysPerf Reporting

Visualization and reporting infrastructure for AgentSysPerf benchmark results.

## Available Report Generators

### XeonPowerPointGenerator

Generates an 8-12 slide PowerPoint presentation highlighting Xeon EMR performance on Terminal-Bench workloads.

**Features:**
- Executive summary with pass/fail rates and workload distribution
- Time breakdown by workload type (inference/execution/orchestration)
- CPU bottleneck classification (io_bound, core_bound, memory_bound, frontend_starved)
- Cache behavior analysis (L3 residency, miss rates)
- Task-level detail tables (5 tasks per slide)
- Aggregated recommendations from all analyzers

**Usage:**

```python
from pathlib import Path
from src.reporting import XeonPowerPointGenerator
from src.storage import SQLiteResultStore

# Initialize store and generator
store = SQLiteResultStore(output_dir=Path("/tmp/results"))
generator = XeonPowerPointGenerator()

# Generate report
output_path = generator.generate_report(
    run_id="run_20260528",
    store=store,
    output_path=Path("/tmp/report.pptx")
)

print(f"Report saved to: {output_path}")
```

**CLI Example:**

```bash
# Run demo with test data
python examples/demo_sqlite_store.py
python examples/demo_pptx_report.py \
    --run-id demo_run_1748445123 \
    --store-path /tmp/agentsysperf_scratch/sqlite_demo \
    --output-dir /tmp/agentsysperf_reports
```

**Report Structure:**

1. **Title Slide** — Run metadata, total tasks, pass rate, workload distribution
2. **Breakdown Chart** — Grouped bar chart of inference/execution/orchestration time by workload type
3. **CPU Classification** — Stacked bar chart showing CPU bottleneck distribution
4. **Cache Behavior** — Dual-axis chart with cache miss % and LLC miss/s
5. **Task Tables** — Paginated tables (5 tasks per slide) with task_id, workload, status, duration, verdicts, IPC
6. **Recommendations** — Aggregated recommendations by theme (model selection, memory optimization, SKU selection, etc.)

**Data Requirements:**

The generator queries data from a `ResultStore` (typically `SQLiteResultStore`):
- `query_tasks(run_id)` — Fetch all tasks for the run
- `query_verdicts(run_id, analyzer_name)` — Fetch analyzer verdicts (breakdown, cpu_bound, cache)

Ensure the following analyzers have been run:
- `breakdown` — Time allocation by inference/execution/orchestration
- `cpu_bound` — CPU bottleneck classification
- `cache` — Cache behavior analysis

**Styling:**

The generator uses Intel brand colors:
- Primary Blue: `#0071C5`
- Cyan: `#00C7FD`
- Orange: `#F0AB00`
- Gray: `#5B6770`

Charts are generated at 150 DPI for publication quality.

## Extending

To add a new report generator:

1. Implement the `ReportGenerator` Protocol from `src.protocols`
2. Add the class to `src/reporting/__init__.py`
3. Register via entry point in `pyproject.toml` (optional for external packages)

**Protocol:**

```python
@runtime_checkable
class ReportGenerator(Protocol):
    name: str
    output_formats: frozenset  # e.g., frozenset(["html", "pdf"])
    
    def generate_report(
        self,
        *,
        run_id: str,
        store: ResultStore,
        output_path: Path,
    ) -> Path:
        """Generate a report and return the output path."""
        ...
```

## Dependencies

- `python-pptx>=1.0.0` — PowerPoint generation
- `matplotlib>=3.5` — Chart rendering
- `seaborn>=0.11` — Enhanced styling
- `pandas>=1.3` — Data manipulation

All dependencies are declared in `pyproject.toml`.
