# Plugin Development Guide — AgentSysPerf 1.0

AgentSysPerf is pluggable. External developers can ship benchmarks, measurements, analyzers, and optimization profiles without forking. This guide covers the entry-point system, plugin registration, and template examples.

Every snippet below is written against `src/protocols.py`, the source of truth for the contracts. In the 1.0 releases the package is imported as `src` (`from src.protocols import ...`).

## Plugin Architecture Overview

AgentSysPerf uses **Python entry points** for automatic plugin discovery. When you run `agentsysperf list`, the CLI enumerates every registered plugin across these groups:

```text
agentsysperf.benchmarks              # BenchmarkAdapter implementations
agentsysperf.measurements            # Measurement implementations
agentsysperf.analyzers               # Analyzer implementations
agentsysperf.optimization_profiles   # OptimizationProfilePlugin implementations
agentsysperf.hardware_telemetry      # HardwareTelemetryPlugin implementations
agentsysperf.result_stores           # ResultStore implementations
agentsysperf.report_generators       # ReportGenerator implementations
```

**No fork required.** Your package just needs:
1. A class implementing one of the Protocols (defined in `src/protocols.py`)
2. An entry-point registration in `pyproject.toml`
3. A `pip install` of your package into the same venv

Discovery instantiates each plugin class with **no arguments** (`cls()`), so a plugin's `__init__` must not require any. Result stores are the exception: they are built with their `open()` classmethod.

---

## Quickstart: Write Your First Analyzer

### Step 1: Understand the Analyzer Protocol

```python
from src.protocols import AnalysisResult


class MyAnalyzer:
    """Your custom analyzer."""

    name: str = "my_analyzer"  # Unique ID (lowercase, no spaces)
    input_layers: frozenset = frozenset(["l1", "l3"])  # Which measurements it consumes

    def analyze(self, records, *, context=None):
        """Pure function: records → verdicts.

        Args:
            records: Sequence of MeasurementRecord objects
            context: Optional AnalysisContext (hardware SKU, baseline, profile)

        Yields:
            AnalysisResult objects (one per span or one aggregate)
        """
        for record in records:
            if record.layer not in self.input_layers:
                continue

            # Your logic: extract features, compute verdict, yield result
            payload = record.payload
            verdict = "your_verdict"
            confidence = 0.92
            evidence = {"key": payload.get("some_metric")}
            recommendations = ["Action 1", "Action 2"]

            yield AnalysisResult(
                span_id=record.span_id,
                verdict=verdict,
                confidence=confidence,
                evidence=evidence,
                recommendations=recommendations,
                analyzer_name=self.name,
            )
```

### Step 2: Create a Package

```
my_agentsysperf_plugin/
├── pyproject.toml
└── my_plugin/
    └── __init__.py
```

`my_plugin/__init__.py`:
```python
# my_plugin/__init__.py
class MyAnalyzer:
    name = "my_analyzer"
    input_layers = frozenset(["l1"])

    def analyze(self, records, *, context=None):
        # Stub
        return []
```

`pyproject.toml`:
```toml
[project]
name = "my-agentsysperf-plugin"
version = "0.1.0"
description = "My custom analyzer for AgentSysPerf"
dependencies = [
    "agentsysperf>=1.0",
]

[project.entry-points."agentsysperf.analyzers"]
my_analyzer = "my_plugin:MyAnalyzer"
```

### Step 3: Install & Discover

```bash
cd my_agentsysperf_plugin
pip install -e .

agentsysperf analyzers list
# The table gains a row:  my_analyzer │ l1 │ MyAnalyzer │ my_plugin
```

---

## Plugin Types & Examples

The examples below live in your own package, `my_pkg`; the first line of each
snippet names its file.

### 1. Writing an Analyzer

```python
# my_pkg/my_analyzer.py
"""MyAnalyzer — detects your custom bottleneck."""

from typing import Dict, Iterable, Optional, Sequence

from src.protocols import AnalysisContext, AnalysisResult, MeasurementRecord


class MyAnalyzer:
    """Classifies workload as type_a or type_b based on feature ratios."""

    name = "my_analyzer"
    input_layers = frozenset(["l3", "l1"])

    def analyze(
        self,
        records: Sequence[MeasurementRecord],
        *,
        context: Optional[AnalysisContext] = None,
    ) -> Iterable[AnalysisResult]:
        """Yield verdicts for each span."""

        # Group records by span
        by_span: Dict[str, Dict[str, MeasurementRecord]] = {}
        for record in records:
            by_span.setdefault(record.span_id, {})[record.layer] = record

        # Analyze each span
        for span_id, layers in by_span.items():
            l3_record = layers.get("l3")
            if not l3_record:
                continue  # Skip spans missing required layers

            # Extract features
            ipc = l3_record.payload.get("ipc")
            cache_miss = l3_record.payload.get("cache_miss_pct")
            if ipc is None or cache_miss is None:
                continue  # Not measured on this span; don't guess

            # Classify
            if ipc < 1.0 and cache_miss > 50:
                verdict = "type_a"
            elif ipc > 2.0 and cache_miss < 20:
                verdict = "type_b"
            else:
                verdict = "type_unknown"

            yield AnalysisResult(
                span_id=span_id,
                verdict=verdict,
                confidence=0.85,
                evidence={
                    "ipc": ipc,
                    "cache_miss_pct": cache_miss,
                    "classification_rule": "rule_xyz",
                },
                recommendations=(
                    ["Recommendation for type_a"] if verdict == "type_a" else []
                ),
                analyzer_name=self.name,
            )
```

**Register in `pyproject.toml`:**

```toml
[project.entry-points."agentsysperf.analyzers"]
my_analyzer = "my_pkg.my_analyzer:MyAnalyzer"
```

---

### 2. Writing a Measurement

A measurement has a `name` and a `layer` (the tag on every record it emits,
which analyzers select on), and four lifecycle methods. AgentSysPerf creates
one instance per run and calls them in this order:

1. `start(*, run_id, output_dir)` once, when the run begins;
2. `observe_span(*, span_id, kind, node_id)` when each span opens;
3. `finalize_span(span_id)` when that span closes; it returns the span's records;
4. `stop()` once, at the end of the run; it returns any run-wide records.

```python
# my_pkg/my_measurement.py
"""MyMeasurement — per-span delta of a system counter."""

from pathlib import Path
from typing import Any, Dict, Iterable

from src.protocols import MeasurementRecord


class MyMeasurement:
    """Collects custom metrics during benchmark spans."""

    name = "my_measurement"   # must match the entry-point name
    layer = "my_layer"        # tag on every record this probe emits

    def __init__(self) -> None:
        self._run_id = ""
        self._open: Dict[str, Dict[str, Any]] = {}

    def start(self, *, run_id: str, output_dir: Path) -> None:
        """Called once when the run begins."""
        self._run_id = run_id

    def observe_span(self, *, span_id: str, kind: str, node_id: str) -> None:
        """Called when a span opens: take the baseline."""
        self._open[span_id] = self._read_system_state()

    def finalize_span(self, span_id: str) -> Iterable[MeasurementRecord]:
        """Called when the span closes: return zero or more records for it."""
        start = self._open.pop(span_id, None)
        if start is None:
            return []
        end = self._read_system_state()
        # MeasurementRecord is a frozen dataclass with exactly these 3 fields.
        return [MeasurementRecord(
            span_id=span_id,
            layer=self.layer,
            payload={
                "my_metric_1": end["metric1"] - start["metric1"],
                "my_metric_2": end["metric2"],
            },
        )]

    def stop(self) -> Iterable[MeasurementRecord]:
        """Called once at the end of the run: return any run-wide records."""
        return []

    def _read_system_state(self) -> Dict[str, Any]:
        """Example: read from /proc or system command."""
        # Your implementation: call APIs, parse files, run commands
        return {"metric1": 100, "metric2": 200}
```

**Register in `pyproject.toml`:**

```toml
[project.entry-points."agentsysperf.measurements"]
my_measurement = "my_pkg.my_measurement:MyMeasurement"
```

**Integration:** Once registered, your measurement will fire on every run (if the system has access to the required resources). Benchmark results will include records in your `layer`.

---

### 3. Writing a Benchmark Adapter

```python
# my_pkg/my_benchmark.py
"""MyBenchmark — custom workload adapter."""

from typing import Any, Callable, Dict, Iterable, Optional, Sequence

from src.protocols import AgentInvoker, TaskResult, TaskSpec

_TASKS = [
    TaskSpec(
        id="task_1",
        instruction="Solve problem A",
        category="reasoning",
        difficulty="easy",
        cpu_budget=1,
        memory_mb=2048,
        timeout_s=300,
    ),
    TaskSpec(
        id="task_2",
        instruction="Solve problem B",
        category="reasoning",
        difficulty="hard",
        cpu_budget=2,
        memory_mb=4096,
        timeout_s=600,
    ),
]


class MyBenchmark:
    """Adapter for your custom benchmark suite."""

    name = "my_benchmark"   # must match the entry-point name
    version = "0.1.0"       # adapter version, not the dataset version

    def list_tasks(
        self,
        *,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> Iterable[TaskSpec]:
        """Yield the tasks that pass the filters, at most `limit` of them."""
        count = 0
        for task in _TASKS:
            if include is not None and task.id not in include:
                continue
            if exclude is not None and task.id in exclude:
                continue
            if limit is not None and count >= limit:
                return
            count += 1
            yield task

    def run_task(
        self,
        task: TaskSpec,
        *,
        agent_invoker: AgentInvoker,
        on_step: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> TaskResult:
        """Execute a task and return its verdict.

        `agent_invoker` is supplied by the runner; delegate to it rather than
        creating your own LLM client. `on_step`, if given, is called per turn.
        """
        try:
            response = agent_invoker.invoke(
                task.instruction,
                metadata={"category": task.category},
                session_hint=task.id,
            )
            passed = self._grade(response)
            return TaskResult(
                task_id=task.id,
                passed=passed,
                reward=1.0 if passed else 0.0,
            )
        except Exception as e:
            return TaskResult(
                task_id=task.id,
                passed=False,
                reward=0.0,
                error=str(e),
            )

    def teardown(self) -> None:
        """Release containers, temp dirs, sandboxes. Nothing to release here."""

    def _grade(self, response: Any) -> bool:
        """Check if the response is correct."""
        # Your grading logic
        return "success" in str(response).lower()
```

**Register in `pyproject.toml`:**

```toml
[project.entry-points."agentsysperf.benchmarks"]
my_benchmark = "my_pkg.my_benchmark:MyBenchmark"
```

**Integration:** a registered adapter is listed by `agentsysperf benchmarks list`.
`agentsysperf run` drives only the built-in `synthetic_cpu` and `terminal-bench`
adapters today, so drive yours from Python, the way the runner does. Open the
measurement span around `run_task`, not around `agent_invoker.invoke`:

```python
# run_my_benchmark.py
from src.measurements.l1_subspan import L1SubSpanMeasurement
from src.runner import RunContext, track_span
from src.testing.noop_invoker import NoOpAgentInvoker  # stand-in; use your AgentInvoker

from my_pkg.my_benchmark import MyBenchmark

adapter = MyBenchmark()
invoker = NoOpAgentInvoker()

# Add more Measurement instances here, or every installed one with
# list(src.protocols.discover_measurements().values()).
with RunContext(measurements=[L1SubSpanMeasurement()]) as ctx:
    for task in adapter.list_tasks(limit=2):
        with track_span(ctx, f"{ctx.run_id}::{task.id}", kind="my_benchmark", node_id=task.id):
            result = adapter.run_task(task, agent_invoker=invoker)
        print(task.id, "passed" if result.passed else "failed")
adapter.teardown()

print(len(ctx.records), "measurement records in", ctx.output_dir)
```

---

### 4. Writing an Optimization Profile

A profile has a `name`, an `is_baseline` flag (True only for `base`), and two
methods. `apply()` projects the profile onto a SUT configuration and returns
the new configuration; it raises when the host cannot satisfy the profile.
`verify_engaged()` reads counters through a `HardwareTelemetryPlugin` and
returns the engagement verdict.

```python
# my_pkg/my_profile.py
"""MyProfile — keep the SUT on one NUMA node, and prove it stayed there."""

from typing import Any, Dict

from src.protocols import HardwareTelemetryPlugin


class MyProfile:
    """Pins the SUT to NUMA node 0."""

    name = "my_profile"
    is_baseline = False

    # Semantic metric names (not raw PMU event names) -> minimum value. A
    # telemetry plugin has to advertise and compute exactly these names.
    verify_counters = {"local_dram_access_ratio": 0.9}

    def apply(self, *, sut_config: Dict[str, Any]) -> Dict[str, Any]:
        """Return a copy of the SUT config with this profile's choices layered in."""
        out = dict(sut_config)
        out["numactl"] = ["--cpunodebind=0", "--membind=0"]
        return out

    def verify_engaged(
        self,
        *,
        telemetry: HardwareTelemetryPlugin,
        run_id: str,
        warmup_done: bool = True,
    ) -> Dict[str, Any]:
        """Read the counters and prove the claimed optimization ran."""
        missing = [c for c in self.verify_counters if c not in telemetry.available_events]
        if missing:
            # Abort, don't degrade: an unmeasurable claim is not "engaged".
            raise RuntimeError(f"{self.name}: telemetry does not advertise {missing}")

        reading = telemetry.read_counters(list(self.verify_counters), window_s=1.0)
        failures = []
        for counter, minimum in self.verify_counters.items():
            value = reading.values.get(counter)
            if value is None or value < minimum:
                failures.append(f"{counter}={value} below min {minimum}")
        return {
            "engaged": not failures,
            "counters": dict(reading.values),
            "thresholds": dict(self.verify_counters),
            "failures": failures,
        }
```

**Register in `pyproject.toml`:**

```toml
[project.entry-points."agentsysperf.optimization_profiles"]
my_profile = "my_pkg.my_profile:MyProfile"
```

**Integrity principle:** "Measured not assumed." Every claim (`name="my_profile"` keeps memory node-local) must be verified by `verify_engaged()` reading actual counters. If a counter is not advertised by telemetry, raise an error (abort-don't-degrade).

`agentsysperf run` does not apply or verify profiles yet; `examples/run_profile_verify.py` exercises `verify_engaged()` against the installed telemetry.

---

## Plugin Discovery & Error Handling

### Auto-discovery

```bash
agentsysperf list
# Enumerates all entry-point plugins; if one fails to import, it's logged (not raised)
```

### Debugging a plugin

If your plugin doesn't appear:

```bash
# A plugin that fails to load prints a warning and its traceback on stderr
agentsysperf analyzers list

# Check sys.path includes your package
python -c "import my_pkg; print(my_pkg.__file__)"

# Verify entry-point registration
pip show -f my_agentsysperf_plugin
```

### Error isolation

A plugin that crashes on import is **logged and skipped**, not raised. This allows third-party plugins to fail without breaking the CLI.

```
AgentSysPerf could not load agentsysperf.analyzers plugin 'my_analyzer' — skipping
Traceback (most recent call last):
  ...
```

---

## Testing Your Plugin

### Unit test template

```python
# tests/test_my_analyzer.py

from src.protocols import MeasurementRecord
from my_pkg.my_analyzer import MyAnalyzer


def test_my_analyzer_classifies_correctly():
    """Verify verdict logic."""
    analyzer = MyAnalyzer()

    # Create a record. MeasurementRecord has exactly span_id, layer, payload.
    record = MeasurementRecord(
        span_id="span_1",
        layer="l3",
        payload={"ipc": 0.8, "cache_miss_pct": 60},
    )

    # Run analysis
    verdicts = list(analyzer.analyze([record]))

    assert len(verdicts) == 1
    assert verdicts[0].verdict == "type_a"
    assert verdicts[0].confidence > 0.5
```

### Integration test

```bash
# Install your plugin
pip install -e my_agentsysperf_plugin

# Run a benchmark and verify your plugin fires. The run prints
# "analyzers N/M emitted (...)" and names any analyzer that stayed silent.
agentsysperf run --benchmark synthetic_cpu --num-tasks 1

# The report's "Analyzer verdicts" table lists every verdict, yours included
agentsysperf report <run_id> --format md --out report.md
grep my_analyzer report.md
```

---

## Protocol Contracts

All Protocols are **SemVer-stable as of 0.1.0**. Signatures will not break at MINOR versions. Adding optional fields is allowed; removing or retyping required ones is not.

### Classes & Required Methods

| Protocol | Module | Required Attributes | Required Methods |
|----------|--------|----------------------|------------------|
| `Analyzer` | `src.protocols` | `name: str`, `input_layers: frozenset` | `analyze(records, *, context=None)` |
| `Measurement` | `src.protocols` | `name: str`, `layer: str` | `start(*, run_id, output_dir)`, `observe_span(*, span_id, kind, node_id)`, `finalize_span(span_id)`, `stop()` |
| `BenchmarkAdapter` | `src.protocols` | `name: str`, `version: str` | `list_tasks(*, include=None, exclude=None, limit=None)`, `run_task(task, *, agent_invoker, on_step=None)`, `teardown()` |
| `OptimizationProfilePlugin` | `src.protocols` | `name: str`, `is_baseline: bool` | `apply(*, sut_config)`, `verify_engaged(*, telemetry, run_id, warmup_done=True)` |
| `HardwareTelemetryPlugin` | `src.protocols` | `name: str`, `available_events: frozenset` | `read_counters(events, *, window_s=1.0, target=None)` |

See `src/protocols.py` for full signatures, including `AgentInvoker`, `ResultStore` and `ReportGenerator`.

---

## Gotchas & Best Practices

### ✓ Do

- **Read the input layers you declare.** If `input_layers = frozenset(["l3"])`, filter on it: `analyze()` receives all of a run's records, so skip layers you did not declare.
- **Return early if layers are missing.** Don't raise; yield no result.
- **Set `analyzer_name=self.name` on every `AnalysisResult`.** The store keys verdicts by run, span and analyzer name, and the report's Analyzer column shows it.
- **Make analyzers stateless.** `analyze()` should be a pure function; no side effects.
- **Measure, don't assume.** In `verify_engaged()`, actually read counters; don't assume config took effect.
- **Use sensible confidence scores.** 0.9+ = very confident; 0.5–0.7 = boundary case; 0.3–0.5 = uncertain.
- **Link to references.** Document thresholds and rationale in docstrings (e.g., "LLC miss > 10M/s based on Yasin 2014").

### ✗ Don't

- **Don't raise in `analyze()`.** Yield empty or low-confidence verdict instead.
- **Don't assume all layers are present.** Defensive code: check before accessing.
- **Don't hardcode paths or system commands** without fallback; use `src.platform` for portability.
- **Don't store mutable state in Analyzer.** Each call to `analyze()` should be independent.
- **Don't forget to register in `pyproject.toml`.** Entry points are the discovery mechanism.

---

## Real-World Example: Latency Analyzer

Here's a complete, working example of a measurement + analyzer pair:

**my_pkg/latency_measurement.py:**
```python
# my_pkg/latency_measurement.py
import time
from pathlib import Path
from typing import Dict, Iterable

from src.protocols import MeasurementRecord


class LatencyMeasurement:
    name = "latency"
    layer = "latency"

    def __init__(self) -> None:
        self._t0: Dict[str, float] = {}

    def start(self, *, run_id: str, output_dir: Path) -> None:
        self._t0.clear()

    def observe_span(self, *, span_id: str, kind: str, node_id: str) -> None:
        self._t0[span_id] = time.monotonic()

    def finalize_span(self, span_id: str) -> Iterable[MeasurementRecord]:
        t0 = self._t0.pop(span_id, None)
        if t0 is None:
            return []
        return [MeasurementRecord(
            span_id=span_id,
            layer=self.layer,
            payload={"elapsed_s": time.monotonic() - t0},
        )]

    def stop(self) -> Iterable[MeasurementRecord]:
        return []
```

**my_pkg/latency_analyzer.py:**
```python
# my_pkg/latency_analyzer.py
from src.protocols import AnalysisResult


class LatencyAnalyzer:
    name = "latency_anomaly"
    input_layers = frozenset(["latency"])

    def analyze(self, records, *, context=None):
        latencies = [r.payload["elapsed_s"] for r in records if r.layer == "latency"]
        if not latencies:
            return

        mean_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)

        if max_latency > 2 * mean_latency:
            yield AnalysisResult(
                span_id=None,
                verdict="latency_anomaly",
                confidence=0.8,
                evidence={"mean": mean_latency, "max": max_latency},
                recommendations=["Investigate tail latency; tune GC settings"],
                analyzer_name=self.name,
            )
```

**pyproject.toml:**
```toml
[project.entry-points."agentsysperf.measurements"]
latency = "my_pkg.latency_measurement:LatencyMeasurement"

[project.entry-points."agentsysperf.analyzers"]
latency_anomaly = "my_pkg.latency_analyzer:LatencyAnalyzer"
```

---

## Troubleshooting

**"My plugin doesn't appear in `agentsysperf list`"**

1. Check `pip list | grep my_pkg` — is it installed?
2. Run `python -c "from my_pkg import MyAnalyzer; print(MyAnalyzer.name)"` — does it import?
3. Check `pyproject.toml` entry-point group name — is it exactly `agentsysperf.analyzers` (for analyzers)?
4. Re-run `pip install -e .` after editing `pyproject.toml`.

**"Plugin loads but analyze() returns nothing"**

1. Check `input_layers` — does your span have those layers? Run `agentsysperf db show <run_id>` to verify.
2. Check if-conditions in `analyze()` — are they too restrictive? Add debug logging.
3. Run with `-v` (verbose) to see INFO-level diagnostics.

**"verify_engaged() never reports engaged"**

1. Ensure the profile actually applied (check `/proc`, `/sys`, or system commands).
2. Verify telemetry has access to the counter you're reading. Some counters need root or `perf_event_paranoid <= 1`.
3. Log the `counters` and `failures` it returns to see what values were actually read.

---

## References

- **Protocols:** `src/protocols.py` (canonical source)
- **Design rationale:** `docs/methodology/DESIGN.md`
- **Reference implementations:**
  - Analyzer: `src/analyzers/cpu_bound.py`
  - Measurement: `src/measurements/l1_subspan/`
  - Benchmark: `src/benchmarks/synthetic_cpu/`
- **Integrity principles:** README, *Integrity principles*
