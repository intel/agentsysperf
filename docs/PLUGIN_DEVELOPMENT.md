# Plugin Development Guide — AgentSysPerf 0.1.0

AgentSysPerf is pluggable. External developers can ship benchmarks, measurements, analyzers, and optimization profiles without forking. This guide covers the entry-point system, plugin registration, and template examples.

## Plugin Architecture Overview

AgentSysPerf uses **Python entry points** for automatic plugin discovery. When you run `agentsysperf list`, the CLI enumerates every registered plugin across these groups:

```python
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

---

## Quickstart: Write Your First Analyzer

### Step 1: Understand the Analyzer Protocol

```python
from src.protocols import Analyzer, AnalysisResult, MeasurementRecord

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
from src.protocols import AnalysisResult, MeasurementRecord

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
    "agentsysperf>=0.1.0",
]

[project.entry-points."agentsysperf.analyzers"]
my_analyzer = "my_plugin:MyAnalyzer"
```

### Step 3: Install & Discover

```bash
cd my_agentsysperf_plugin
pip install -e .

agentsysperf analyzers list
# Output includes: my_analyzer (input_layers: l1)
```

---

## Plugin Types & Examples

### 1. Writing an Analyzer

**File:** `src/analyzers/my_analyzer.py`

```python
"""MyAnalyzer — detects your custom bottleneck."""

from typing import Iterable, Optional, Sequence
from src.protocols import AnalysisResult, AnalysisContext, MeasurementRecord

class MyAnalyzer:
    """Classifies workload as type_A or type_B based on feature ratios."""
    
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
        by_span = {}
        for record in records:
            if record.span_id not in by_span:
                by_span[record.span_id] = {}
            by_span[record.span_id][record.layer] = record
        
        # Analyze each span
        for span_id, layers in by_span.items():
            l3_record = layers.get("l3")
            l1_record = layers.get("l1")
            
            if not l3_record:
                continue  # Skip spans missing required layers
            
            # Extract features
            ipc = l3_record.payload.get("ipc")
            cache_miss = l3_record.payload.get("cache_miss_pct")
            
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
                recommendations=[
                    "Recommendation for type_a" if verdict == "type_a" else "",
                ],
            )
```

**Register in `pyproject.toml`:**

```toml
[project.entry-points."agentsysperf.analyzers"]
my_analyzer = "my_pkg.my_analyzer:MyAnalyzer"
```

---

### 2. Writing a Measurement

**File:** `src/measurements/my_measurement.py`

```python
"""MyMeasurement — custom hardware probe."""

from dataclasses import dataclass
from typing import Any, Dict

from src.protocols import Measurement, MeasurementRecord
from src.runner import RunContext

@dataclass
class MyMeasurementRecord(MeasurementRecord):
    """Concrete record type for MyMeasurement."""
    pass

class MyMeasurement:
    """Collects custom metrics during benchmark spans."""
    
    name = "my_measurement"
    
    def __init__(self, context: RunContext):
        """Initialize the probe with access to run metadata."""
        self.context = context
    
    def start_span(self, span_id: str, **metadata) -> None:
        """Called when a span begins."""
        # E.g., read baseline system state
        self.span_start_state = self._read_system_state()
    
    def end_span(self, span_id: str) -> MeasurementRecord:
        """Called when a span ends. Return a measurement record or None."""
        # E.g., compute delta from start to end
        end_state = self._read_system_state()
        delta = {
            "my_metric_1": end_state["metric1"] - self.span_start_state["metric1"],
            "my_metric_2": end_state["metric2"],
        }
        
        return MyMeasurementRecord(
            run_id=self.context.run_id,
            span_id=span_id,
            layer=self.name,
            payload=delta,
        )
    
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

**File:** `src/benchmarks/my_benchmark.py`

```python
"""MyBenchmark — custom workload adapter."""

from typing import List, Optional
from src.protocols import BenchmarkAdapter, TaskSpec, TaskResult

class MyBenchmark:
    """Adapter for your custom benchmark suite."""
    
    name = "my_benchmark"
    
    def __init__(self, context=None):
        """Initialize. Context may include CLI args or config."""
        self.context = context
    
    def list_tasks(self) -> List[TaskSpec]:
        """Return all available tasks.
        
        Each TaskSpec includes id, instruction, difficulty, timeout, etc.
        """
        return [
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
    
    def run_task(self, task: TaskSpec, agent_invoker) -> TaskResult:
        """Execute a task and return result.
        
        Args:
            task: One TaskSpec from list_tasks()
            agent_invoker: AgentInvoker (call it to invoke LLM, tools, etc.)
        
        Returns:
            TaskResult with passed (bool), reward (float), error (str)
        """
        try:
            # Invoke agent with task instruction
            response = agent_invoker.invoke(
                task.instruction,
                tools=[],  # your tools here
            )
            
            # Grade response
            passed = self._grade(response, task.id)
            reward = 1.0 if passed else 0.5
            
            return TaskResult(
                task_id=task.id,
                passed=passed,
                reward=reward,
                error=None,
            )
        except Exception as e:
            return TaskResult(
                task_id=task.id,
                passed=False,
                reward=0.0,
                error=str(e),
            )
    
    def _grade(self, response: str, task_id: str) -> bool:
        """Check if response is correct."""
        # Your grading logic
        return "success" in response.lower()
```

**Register in `pyproject.toml`:**

```toml
[project.entry-points."agentsysperf.benchmarks"]
my_benchmark = "my_pkg.my_benchmark:MyBenchmark"
```

**Integration:** Run with:
```bash
agentsysperf run -b my_benchmark --num-tasks 2
```

---

### 4. Writing an Optimization Profile

**File:** `src/optimization_profiles/my_profile.py`

```python
"""MyProfile — a system tuning configuration."""

from src.protocols import OptimizationProfilePlugin, HardwareTelemetryPlugin

class MyProfile:
    """Enables custom CPU/memory/I/O tuning."""
    
    name = "my_profile"
    
    def apply(self, telemetry: HardwareTelemetryPlugin) -> None:
        """Apply the profile (tune OS, CPU, memory)."""
        # Example: set CPU governor, enable turbo, tune NUMA
        # This is pseudo-code; real impl would use subprocess/sysfs
        subprocess.run(["echo", "performance"], check=True)
        subprocess.run(["echo", "1"], check=True)  # enable turbo
    
    def verify_engaged(self, telemetry: HardwareTelemetryPlugin) -> bool:
        """Verify the profile is actually active via telemetry.
        
        Returns True if counters confirm engagement; False otherwise.
        IMPORTANT: Do not assume. Measure.
        """
        # Read counters from telemetry
        counters = telemetry.read_counters()
        
        # Check: is turbo active?
        turbo_freq = counters.get("turbo_freq_mhz")
        if turbo_freq < 3000:
            return False  # Turbo not engaged
        
        return True
```

**Register in `pyproject.toml`:**

```toml
[project.entry-points."agentsysperf.optimization_profiles"]
my_profile = "my_pkg.my_profile:MyProfile"
```

**Integrity principle:** "Measured not assumed." Every claim (`name="my_profile"` enables turbo) must be verified by `verify_engaged()` reading actual counters. If a counter is not advertised by telemetry, raise an error (abort-don't-degrade).

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
# Enable debug logging
agentsysperf list -v

# Check sys.path includes your package
python -c "import my_pkg; print(my_pkg.__file__)"

# Verify entry-point registration
pip show -f my_agentsysperf_plugin
```

### Error isolation

A plugin that crashes on import is **logged and skipped**, not raised. This allows third-party plugins to fail without breaking the CLI.

```
WARNING: Plugin 'my_analyzer' failed to load: ImportError: no module named 'foo'
```

---

## Testing Your Plugin

### Unit test template

```python
# tests/test_my_analyzer.py

import pytest
from src.protocols import MeasurementRecord
from my_pkg.my_analyzer import MyAnalyzer

def test_my_analyzer_classifies_correctly():
    """Verify verdict logic."""
    analyzer = MyAnalyzer()
    
    # Create mock records
    record = MeasurementRecord(
        run_id="test_run",
        span_id="span_1",
        layer="l3",
        payload={"ipc": 0.8, "cache_miss_pct": 60},
    )
    
    # Run analysis
    verdicts = list(analyzer.analyze([record]))
    
    assert len(verdicts) == 1
    assert verdicts[0].verdict == "memory_bound"
    assert verdicts[0].confidence > 0.5
```

### Integration test

```bash
# Install your plugin
pip install -e my_agentsysperf_plugin

# Run a benchmark and verify your plugin fires
agentsysperf run --benchmark synthetic_cpu --num-tasks 1

# Check plugin output
agentsysperf db show <run_id> | grep "my_analyzer"
```

---

## Protocol Contracts

All Protocols are **SemVer-stable as of 0.1.0**. Signatures will not break at MINOR versions. Adding optional fields is allowed; removing or retyping required ones is not.

### Classes & Required Methods

| Protocol | Module | Required Attributes | Required Methods |
|----------|--------|----------------------|------------------|
| `Analyzer` | `src.protocols` | `name: str`, `input_layers: frozenset` | `analyze(records, *, context=None)` |
| `Measurement` | `src.protocols` | `name: str` | `start_span(span_id, **metadata)`, `end_span(span_id)` |
| `BenchmarkAdapter` | `src.protocols` | `name: str` | `list_tasks()`, `run_task(task, agent_invoker)` |
| `OptimizationProfilePlugin` | `src.protocols` | `name: str` | `apply(telemetry)`, `verify_engaged(telemetry)` |

See `src/protocols.py` for full signatures.

---

## Gotchas & Best Practices

### ✓ Do

- **Read the input layers you declare.** If `input_layers = frozenset(["l3"])`, your analyzer will only receive L3 records.
- **Return early if layers are missing.** Don't raise; yield no result.
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

**measurement.py:**
```python
from dataclasses import dataclass
import time
from src.protocols import MeasurementRecord

@dataclass
class LatencyRecord(MeasurementRecord):
    pass

class LatencyMeasurement:
    name = "latency"
    
    def __init__(self, context):
        self.context = context
        self.span_times = {}
    
    def start_span(self, span_id, **metadata):
        self.span_times[span_id] = time.time()
    
    def end_span(self, span_id):
        elapsed = time.time() - self.span_times.pop(span_id, time.time())
        return LatencyRecord(
            run_id=self.context.run_id,
            span_id=span_id,
            layer="latency",
            payload={"elapsed_s": elapsed},
        )
```

**analyzer.py:**
```python
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
            )
```

**pyproject.toml:**
```toml
[project.entry-points."agentsysperf.measurements"]
latency = "my_pkg:LatencyMeasurement"

[project.entry-points."agentsysperf.analyzers"]
latency_anomaly = "my_pkg:LatencyAnalyzer"
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

**"verify_engaged() always returns False"**

1. Ensure the profile actually applied (check `/proc`, `/sys`, or system commands).
2. Verify telemetry has access to the counter you're reading. Some counters need root or `perf_event_paranoid <= 1`.
3. Add logging to `verify_engaged()` to see what counter value you're actually reading.

---

## References

- **Protocols:** `src/protocols.py` (canonical source)
- **Design rationale:** `docs/methodology/DESIGN.md`
- **Reference implementations:**
  - Analyzer: `src/analyzers/cpu_bound.py`
  - Measurement: `src/measurements/l1_subspan.py`
  - Benchmark: `src/benchmarks/synthetic_cpu.py`
- **Integrity principles:** README, *Integrity principles*
