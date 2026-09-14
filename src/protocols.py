#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""
AgentSysPerf v1 plugin Protocols  —  the contract every benchmark, measurement,
optimization profile, and hardware-telemetry plugin writes to.

This module defines the **four load-bearing Protocols** that govern how
external developers extend AgentSysPerf:

- :class:`BenchmarkAdapter` — drives a workload (Terminal-Bench, SWE-Bench,
  tau-bench, synthetic CPU, your own benchmark) through a system-under-test.
  Each adapter is one folder under ``src.benchmarks.<name>``.
- :class:`Measurement` — a probe that emits records during a benchmark run.
  L1 (sub-spans) is the canonical reference. L3 (perf-stat / TMA),
  L5 (PCM / RAPL / eBPF), VTune are plugins that follow the same lifecycle.
  Each measurement is one folder under ``src.measurements.<name>``.
- :class:`OptimizationProfilePlugin` — owns one named optimization profile
  (``base``, ``amx_only``, ``full_xeon``, ``openvino_xeon``, ...). Applies
  the profile to a SUT, then **verifies engagement** — no claim is trusted
  by config alone; every claim is gated on counter activity.
- :class:`HardwareTelemetryPlugin` — counter source (perf, PCM, RAPL,
  dcgmi, ...). Doubles as the optimization-engagement verifier consumed
  by :class:`OptimizationProfilePlugin`. Per the design's integrity
  principle, "Built with AMX" is not proof AMX ran.

Plugin discovery uses Python entry points (``agentsysperf.benchmarks``,
``agentsysperf.measurements``, ``agentsysperf.optimization_profiles``,
``agentsysperf.hardware_telemetry``).  External packages can ship plugins
without forking AgentSysPerf — pip install gets you the plugin.

Compatibility: these Protocols are SemVer-stable as of 0.1.0. Signatures
will not break at MINOR. Adding optional fields is allowed; removing or
retyping required ones is not.

Methodology references:

- ``docs/methodology/DESIGN.md`` — the 16-section design document.
- ``README.md``, *Integrity principles* — the rules every plugin honors
  (measured-not-assumed, verify-don't-trust, abort-don't-degrade, and no
  fabricated PMU codes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)


# =============================================================================
# Lightweight value types
# =============================================================================


@dataclass(frozen=True)
class TaskSpec:
    """One unit of work a :class:`BenchmarkAdapter` can run.

    Carries the inputs the adapter needs to instantiate its environment
    and dispatch the agent against it.  ``id`` must be stable across
    runs of the same dataset — analyzers and reporters key on it.

    Adapters are free to subclass or extend ``extra`` for their own
    state, but the listed fields are what AgentSysPerf's CLI and reporters
    rely on.
    """

    id: str
    instruction: str
    category: str = ""
    difficulty: str = ""
    cpu_budget: int = 1
    memory_mb: int = 2048
    timeout_s: float = 900.0
    expects_internet: bool = False
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskResult:
    """The verdict for one :class:`TaskSpec` after the agent runs.

    ``passed`` is the binary pass/fail used by the leaderboard story.
    ``reward`` is the canonical numeric score (0.0–1.0 typically) so
    benchmarks with graded judging can express partial credit.
    ``trajectory_path`` lets reporters slurp per-turn detail without
    every adapter having to invent its own format.
    """

    task_id: str
    passed: bool
    reward: float = 0.0
    error: Optional[str] = None
    trajectory_path: Optional[Path] = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MeasurementRecord:
    """One row emitted by a :class:`Measurement` plugin.

    A measurement attaches to a run via :meth:`Measurement.start` and
    emits zero or more records keyed by ``span_id`` (the span this
    record annotates) plus a ``layer`` tag that identifies which probe
    produced it.

    The ``payload`` is intentionally free-form — AgentSysPerf serializes
    it under a namespaced key (``measurements.<layer>.*``) so different
    probes can coexist on the same span without colliding.
    """

    span_id: str
    layer: str
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class CounterReading:
    """A snapshot of hardware counters from :class:`HardwareTelemetryPlugin`.

    Counters are vendor- and SKU-specific; the ``values`` mapping carries
    raw event names (e.g. ``"amx_active_cycles"``, ``"l1d.replacement"``)
    to their accumulated count over the snapshot window. ``window_s`` is
    the wall-clock span the snapshot covers; consumers compute rates by
    dividing.

    The counter source MUST NOT fabricate events that aren't actually
    available on the running CPU — return them omitted, not zero.
    Per the no-fabricated-PMU-codes rule (README, *Integrity
    principles*), fabricated PMU codes are a correctness defect.
    """

    values: Mapping[str, float]
    window_s: float
    cpu_model: str = ""
    notes: str = ""


@dataclass(frozen=True)
class AnalysisResult:
    """One insight derived from raw :class:`MeasurementRecord` data.

    Analyzers transform low-level counters (IPC, cache misses, RSS) into
    domain-specific insights ("memory-bound", "cache-resident", "leak").
    Each result carries a verdict, confidence score, evidence, and
    actionable recommendations.
    """

    verdict: str
    confidence: float  # 0.0-1.0
    evidence: Mapping[str, Any]
    recommendations: Sequence[str] = field(default_factory=list)
    span_id: Optional[str] = None  # If analyzing one span, its ID
    analyzer_name: str = ""


@dataclass
class AnalysisContext:
    """Optional context for analyzers.

    Carries metadata about the run (hardware SKU, optimization profile,
    baseline for comparison) so analyzers can produce contextual insights.
    """

    hardware_sku: str = ""
    optimization_profile: str = ""
    baseline_records: Sequence[MeasurementRecord] = field(default_factory=list)
    extra: Mapping[str, Any] = field(default_factory=dict)


# =============================================================================
# BenchmarkAdapter — the contract for adding a new benchmark
# =============================================================================


@runtime_checkable
class BenchmarkAdapter(Protocol):
    """Drives a workload through a system-under-test (SUT).

    A benchmark adapter is responsible for:

    1. **Listing tasks** — yield :class:`TaskSpec` from whatever
       upstream source defines them (Harbor cache, JSONL file, HF
       dataset, etc.).
    2. **Running one task** — given a SUT (e.g., a callable that
       invokes the agent under measurement), provision the
       environment, dispatch the agent, and return a
       :class:`TaskResult`.
    3. **Cleanup** — release any container, tmp dir, or sandbox.

    The adapter does NOT know which measurement plugins are active —
    AgentSysPerf attaches measurements via :class:`Measurement.start`
    around the call.  This keeps benchmarks and measurements
    independently composable.

    Required class attributes:

    - ``name`` — short string id used in CLI flags (e.g.
      ``"terminal-bench"``).  Must match the entry-point name.
    - ``version`` — adapter version, not the dataset version.
    """

    name: str
    version: str

    def list_tasks(
        self,
        *,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> Iterable[TaskSpec]:
        """Yield TaskSpec instances matching the include/exclude filters.

        Adapters should respect ``limit`` and stop iteration when the
        cap is reached — saves wall time for smoke runs.
        """
        ...

    def run_task(
        self,
        task: TaskSpec,
        *,
        agent_invoker: "AgentInvoker",
        on_step: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> TaskResult:
        """Run one task end-to-end and return its verdict.

        ``agent_invoker`` is supplied by AgentSysPerf — calling it
        triggers the agent loop with measurements active.  Adapters
        should not instantiate their own LiteLLM client; they delegate
        to the invoker.

        ``on_step`` (optional) is called for each agent turn so
        external observers can stream progress.

        **Span convention (load-bearing — read this).** Runners MUST
        open the per-task measurement span (``Measurement.observe_span``
        / ``finalize_span``) around the *call to this method*, not
        around ``agent_invoker.invoke``. Two reasons:

        1. Some adapters do not call the invoker. The synthetic_cpu
           reference adapter runs CPU workloads in-process and ignores
           ``agent_invoker`` entirely. A runner that brackets spans on
           ``.invoke`` would emit zero measurement records for those
           adapters and look like a broken plugin.
        2. ``run_task`` is the unit a benchmark reports on. Wrapping
           the span here is what makes ``span_id`` line up with the
           ``task.id`` analyzers and reporters key off.

        Adapters MAY open nested sub-spans inside ``run_task`` (per
        agent turn, per LLM phase, per tool call). Nested spans are
        attributed correctly because ``Measurement`` plugins observe
        each ``observe_span`` / ``finalize_span`` pair independently.
        """
        ...

    def teardown(self) -> None:
        """Release any persistent resources (containers, tmp dirs)."""
        ...


# =============================================================================
# AgentInvoker — the SUT handle adapters use to drive the agent
# =============================================================================


@runtime_checkable
class AgentInvoker(Protocol):
    """Opaque handle that lets a :class:`BenchmarkAdapter` invoke the
    agent under measurement, without knowing which framework wraps it.

    AgentSysPerf supplies one of these per run, configured to talk to the
    user's chosen integration path (LangChain ``AgentFlowChatModel``,
    LiteLLM custom-provider, future SDK shims).

    The minimum surface is one method, ``invoke``, which delivers a
    prompt + environment context and returns whatever the agent
    produced.  Adapters interpret the return value with their own
    test oracle.
    """

    def invoke(
        self,
        instruction: str,
        *,
        environment: Any = None,
        metadata: Optional[Mapping[str, Any]] = None,
        session_hint: Optional[str] = None,
    ) -> Any:
        """Run the agent on ``instruction`` with optional env handle.

        Returns whatever the agent produced (final message, dict of
        terminal state, patch, file diff, ...).  The adapter scores it.

        ``session_hint`` is an opaque identifier the adapter can pass
        (typically ``task.id``) to request KV-cache affinity from
        runtimes that support it.  Implementations that don't honor
        sessions ignore this field — adapters can pass it
        unconditionally without checking which invoker is wired in.
        """
        ...


# =============================================================================
# Measurement — the contract for adding a new probe / layer
# =============================================================================


@runtime_checkable
class Measurement(Protocol):
    """A probe that observes a run and emits :class:`MeasurementRecord`.

    Examples:

    - L1 sub-spans (request_build / gateway_call / post_process) — the
      canonical reference plugin.
    - L2 py-spy flame graphs scoped to spans.
    - L3 ``perf stat`` TMA topdown counters scoped to span windows
      (ported from the harness perf_sampler).
    - L4 SMT/scheduler/GIL diagnostics (taskset ablation, perf sched,
      gil_load).
    - L5 socket-level (Intel PCM uncore PMC + RAPL energy + eBPF
      off-CPU stack-traces).
    - VTune integration — Intel's profiler attached at span boundaries.

    A measurement plugin's lifecycle:

    1. ``start()`` is called once at the beginning of a run.  The
       plugin captures whatever baseline state it needs (counter
       baseline, profiler attach, kernel-tracing setup).
    2. ``observe_span(span_id, kind, node_id)`` is called when each
       AgentSysPerf span opens — gives the plugin a chance to bracket
       its measurement (e.g., reset counters, start a per-span
       sample window).
    3. ``finalize_span(span_id)`` is called when the span closes —
       returns zero or more :class:`MeasurementRecord` for that span.
    4. ``stop()`` is called at end of run for global teardown.

    Plugins are registered via the ``agentsysperf.measurements`` entry
    point.  AgentSysPerf instantiates one Measurement instance per run,
    not per span.

    Plugins that produce process-wide records (no span association)
    should still emit them from ``stop()`` keyed to the run's root
    span.

    **Where the per-task span is opened.** Runners open the per-task
    span around :meth:`BenchmarkAdapter.run_task`, not around
    :meth:`AgentInvoker.invoke`. That is the contract every plugin
    can rely on: ``observe_span`` fires before ``run_task`` starts,
    ``finalize_span`` fires after it returns. Adapters MAY open
    nested spans inside ``run_task`` for per-turn or per-phase
    detail; plugins see those independently and emit their own
    records keyed by the inner ``span_id``. See
    :meth:`BenchmarkAdapter.run_task` for the rationale.
    """

    name: str
    layer: str  # "l1", "l2", "l3", "l4", "l5", "vtune", or custom

    def start(self, *, run_id: str, output_dir: Path) -> None:
        """Begin a run.  Capture baseline state.  Idempotent."""
        ...

    def observe_span(
        self,
        *,
        span_id: str,
        kind: str,
        node_id: str,
    ) -> None:
        """Called when a span opens.  Default: no-op."""
        ...

    def finalize_span(self, span_id: str) -> Iterable[MeasurementRecord]:
        """Called when a span closes.  Yield records for the span."""
        ...

    def stop(self) -> Iterable[MeasurementRecord]:
        """End of run.  Yield any process-wide / late records."""
        ...


# =============================================================================
# OptimizationProfilePlugin — owns one configuration arm; verifies engagement
# =============================================================================


@runtime_checkable
class OptimizationProfilePlugin(Protocol):
    """Owns one named optimization profile and proves it engaged.

    A profile is a coordinated set of build flags, runtime ISA choices,
    NUMA / hugepage / isolcpus settings, accelerator offloads, and so
    on. The plugin's job is **not** to be an opaque toggle — it is
    to apply its configuration AND prove on hardware counters that the
    claimed work actually ran.

    This contract enforces two integrity principles from the design:

    - **Measured not assumed**: each profile is a swept variable, not a
      built-in feature. The benchmark must be able to return "AMX
      doesn't help here" as a publishable outcome.
    - **Verify, don't trust the label**: ``apply()`` configures the SUT;
      ``verify_engaged()`` reads counters via a
      :class:`HardwareTelemetryPlugin` and aborts the run if claimed
      optimizations show ~zero activity. "Built with AMX" is not
      evidence AMX ran.

    The reference implementations (``base``, ``amx_only``, ``amx_onednn``,
    ``amx_numa_hugepages``, ``full_xeon``, ``openvino_xeon``) live in
    ``docs/contracts/optimization_profiles.py`` as a methodology
    reference; runtime ports get registered via the
    ``agentsysperf.optimization_profiles`` entry point.

    Required class attributes:

    - ``name`` — short string id (e.g. ``"amx_onednn"``). Matches entry point.
    - ``is_baseline`` — True only for the ``base`` profile. The
      ``baseline_purity_check`` invalidates a run if a baseline profile
      shows AMX or other opt-in counter activity.
    """

    name: str
    is_baseline: bool

    def apply(self, *, sut_config: Dict[str, Any]) -> Dict[str, Any]:
        """Project the profile onto a SUT-specific config.

        Takes the user's SUT config (model, backend, scheduler hints)
        and returns a new config with this profile's choices layered in
        (ISA flags, env vars, numactl wrap, taskset cpulist, hugepage
        mount, accelerator queue config, ...).

        Profiles MUST NOT silently downgrade. If the host can't satisfy
        the profile (e.g. no AMX), apply() raises — the caller skips
        this arm rather than running it as ``base``-equivalent.
        """
        ...

    def verify_engaged(
        self,
        *,
        telemetry: "HardwareTelemetryPlugin",
        run_id: str,
        warmup_done: bool = True,
    ) -> Dict[str, Any]:
        """Read counters and prove the claimed optimizations ran.

        Returns a dict with at minimum:

        - ``engaged`` (bool): all required counter thresholds passed.
        - ``counters`` (dict): the actual readings used, for the report.
        - ``thresholds`` (dict): the thresholds checked, for audit.
        - ``failures`` (list): empty if engaged, else specific reasons.

        Implementations should raise an exception (not return
        ``engaged=False`` silently) when the run cannot continue —
        per ``docs/methodology/DESIGN.md §9``, "abort, don't degrade".

        For the ``base`` profile, this is the *purity check*: it must
        verify that opt-in features (AMX, hugepages, isolcpus) are
        NOT active. The whole study is invalidated if a baseline shows
        them.
        """
        ...


# =============================================================================
# HardwareTelemetryPlugin — counter source + engagement verifier
# =============================================================================


@runtime_checkable
class HardwareTelemetryPlugin(Protocol):
    """Reads hardware performance counters; doubles as engagement verifier.

    Per the design's deliberate choice (``docs/methodology/DESIGN.md §9``), the same
    contract that surfaces telemetry to reports also serves as the
    verifier consumed by :class:`OptimizationProfilePlugin`. There is
    no separate "perf characterization" plugin — verification is just
    counter reads with thresholds.

    Reference implementations target ``perf stat`` (Linux), Intel PCM
    (uncore + RAPL), Nvidia ``dcgmi``, and so on. The first port is
    the L3 perf-stat plugin under ``src/measurements/l3_perf/``,
    which can be used both as a :class:`Measurement` (per-span counter
    rows) and as a :class:`HardwareTelemetryPlugin` (point-in-time
    snapshots for ``verify_engaged``).

    Required class attributes:

    - ``name`` — short string id (e.g. ``"perf_stat"``, ``"intel_pcm"``).
    - ``available_events`` — frozenset of event names this plugin can
      report on the running host. Profiles consult this before asking
      for events that don't exist on the SKU; per the no-fabricated-PMU-codes
      rule (README, *Integrity principles*), fabricated PMU codes
      are a correctness defect.

    Naming contract (load-bearing, and easy to get wrong): this namespace is
    **semantic metric names, not raw PMU event names**. Profiles declare what
    they need in ``SPEC.verify_counters`` using names like
    ``amx_active_cycle_ratio`` or ``hugepage_fault_ratio``; engagement
    verification matches those strings against ``available_events``
    literally. A plugin that advertises only raw perf event names therefore
    satisfies no profile counter, even on hardware where the underlying events
    exist. Whoever wants a profile verified must ship a plugin that advertises
    — and computes, in ``read_counters`` — that profile's metric names.

    Corollary: not every profile counter is a PMU quantity.
    ``hugepage_fault_ratio`` comes from ``/proc/vmstat``,
    ``accel_queue_depth_qat`` from QAT sysfs, and
    ``onednn_kernel_dispatch_ratio`` from library-level instrumentation. A
    vendor PMU plugin (``intel_pcm``) is necessary but not sufficient to make
    the shipped Xeon profiles verifiable; see ``docs/measurement_layers.md``.
    """

    name: str
    available_events: frozenset

    def read_counters(
        self,
        events: Sequence[str],
        *,
        window_s: float = 1.0,
        target: Optional[str] = None,
    ) -> CounterReading:
        """Sample ``events`` for ``window_s`` and return the snapshot.

        ``target`` is optional and plugin-specific: a PID, cgroup path,
        container ID, or socket selector. If omitted, the plugin reads
        system-wide.

        Plugins MUST silently drop events not in
        :attr:`available_events`. The returned ``CounterReading.values``
        only contains events the plugin actually measured.

        Plugins MUST NOT block longer than ``window_s + small_overhead``;
        the caller may invoke this repeatedly inside a span window.
        """
        ...


# =============================================================================
# Analyzer — transforms raw measurements into domain insights
# =============================================================================


@runtime_checkable
class Analyzer(Protocol):
    """Derives insights from raw :class:`MeasurementRecord` data.

    Analyzers are the layer between raw counters and actionable
    recommendations. Examples:

    - CPUBoundAnalyzer: IPC + cache metrics → "core-bound" | "memory-bound"
    - CacheAnalyzer: L1/L2/L3 miss rates → dominant cache tier
    - MemoryLeakAnalyzer: RSS growth → leak detection
    - CPUBurstAnalyzer: CPU% variance → burst vs sustained
    - CacheCoherenceAnalyzer: LLC misses + context switches → thrashing

    Analyzers are **stateless**: :meth:`analyze` is a pure function over
    records.  This enables offline analysis (collect once, analyze many
    times), parallel execution, and caching.

    Multiple analyzers can process the same raw data (CPUBoundAnalyzer +
    CacheAnalyzer both read L3 records).  Composition is automatic via
    entry-point discovery.

    Required class attributes:

    - ``name`` — short string id (e.g. ``"cpu_bound"``).
    - ``input_layers`` — frozenset of :class:`Measurement` layer names
      this analyzer consumes (e.g. ``frozenset(["l1", "l3"])``).
    """

    name: str
    input_layers: frozenset

    def analyze(
        self,
        records: Sequence[MeasurementRecord],
        *,
        context: Optional[AnalysisContext] = None,
    ) -> Iterable[AnalysisResult]:
        """Derive insights from measurement records.

        Called with all records from a run (or a subset filtered by
        span_id).  Returns zero or more :class:`AnalysisResult` instances.

        Analyzers should be defensive: missing expected layers or fields
        should yield empty results or low-confidence verdicts, not raise.

        ``context`` carries optional metadata (hardware SKU, optimization
        profile, baseline for comparison).  Analyzers MAY use it to
        produce contextual insights (e.g., "IPC lower than baseline").
        """
        ...


# =============================================================================
# ResultStore — persists benchmark results for analysis and reporting
# =============================================================================


@runtime_checkable
class ResultStore(Protocol):
    """Stores benchmark results for analysis and reporting.

    A result store persists run metadata, task results, measurement records,
    and analysis results in a queryable backend (SQLite, PostgreSQL, S3 + Athena,
    etc.). This enables offline analysis, historical comparison, and report
    generation without re-running benchmarks.

    The reference implementation is :class:`SQLiteResultStore` in
    ``src.storage.sqlite_store``, which writes to a local SQLite database
    in the output directory. External packages can ship alternative backends
    (PostgreSQL, cloud object storage) via the ``agentsysperf.result_stores``
    entry point.

    Required class attributes:

    - ``name`` — short string id (e.g. ``"sqlite"``).
    """

    name: str

    # ── Construction seam ──────────────────────────────────────────────
    # The single blessed constructor: resolves the canonical local store, or a
    # shared-server backend by DSN, with no caller change. discover_result_stores
    # calls this. Backends register via the ``agentsysperf.result_stores`` entry
    # point and implement ``open()`` + the methods below.
    @classmethod
    def open(cls, *, dsn: Optional[str] = None, read_only: bool = False) -> "ResultStore":
        """Open the canonical store (local file by default, server by DSN)."""
        ...

    # ── Writes ─────────────────────────────────────────────────────────
    def store_run_metadata(self, *, run_id: str, metadata: Dict[str, Any]) -> None:
        """Record run-level metadata (hardware SKU, model, start time, etc.)."""
        ...

    def store_task_result(self, *, run_id: str, task_id: str, result: Dict[str, Any]) -> None:
        """Record the result of one task (pass/fail, duration, turns, etc.)."""
        ...

    def store_measurements(self, *, run_id: str, records: Sequence[MeasurementRecord]) -> None:
        """Persist measurement records (L1/L3/L5/perfspect) for a run."""
        ...

    def store_analysis_results(self, *, run_id: str, results: Sequence[AnalysisResult]) -> None:
        """Persist analyzer verdicts and recommendations."""
        ...

    def store_spans(self, *, run_id: str, spans: Sequence[Any]) -> None:
        """Persist step-level execution-trace rows (LLM/tool/agent steps)."""
        ...

    def persist_run(
        self,
        *,
        run_id: str,
        metadata: Dict[str, Any],
        task_results: Sequence[Any] = (),
        records: Sequence[MeasurementRecord] = (),
        spans: Sequence[Any] = (),
        verdicts: Sequence[AnalysisResult] = (),
        artifacts: Sequence[Dict[str, Any]] = (),
    ) -> None:
        """Atomically flush a whole run (metadata + tasks + measurements +
        spans + verdicts + artifacts) in one transaction.

        ``artifacts`` is a sequence of ``{"kind", "name", "path"}`` dicts."""
        ...

    def store_sweep_metadata(self, *, sweep_id: str, metadata: Dict[str, Any]) -> None:
        """Record sweep-level metadata (one row per concurrency sweep)."""
        ...

    def store_sweep_point(self, *, sweep_id: str, point: Dict[str, Any]) -> None:
        """Record one density cell's rollup for a sweep."""
        ...

    # ── Reads ──────────────────────────────────────────────────────────
    def query_tasks(self, run_id: str, workload_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch task results for a run, optionally filtered by workload type."""
        ...

    def query_verdicts(self, run_id: str, analyzer_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch analyzer verdicts for a run, optionally filtered by analyzer."""
        ...

    def query_measurements(
        self,
        run_id: str,
        *,
        layer: Optional[str] = None,
        span_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch measurement rows for a run; payload decoded, ordered by seq."""
        ...

    def query_spans(
        self, run_id: str, task_id: Optional[str] = None, span_kind: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch step-trace rows for a run, optionally filtered."""
        ...

    def query_sweeps(self) -> List[Dict[str, Any]]:
        """List all sweeps (most recent first)."""
        ...

    def query_sweep_points(self, sweep_id: str) -> List[Dict[str, Any]]:
        """Fetch all density-cell rollups for a sweep, ordered by density."""
        ...


# =============================================================================
# ReportGenerator — renders stored results into human-readable reports
# =============================================================================


@runtime_checkable
class ReportGenerator(Protocol):
    """Generates reports from stored results.

    A report generator consumes :class:`ResultStore` data and produces
    human-readable artifacts (HTML dashboards, PDF summaries, Markdown tables,
    Jupyter notebooks, ...).

    The reference implementation (roadmap Phase 6) will produce an HTML report
    with task pass/fail breakdowns, analyzer verdicts, L1/L3 counter tables,
    and SKU comparison charts. External packages can ship custom formats
    (PDF, LaTeX, interactive Streamlit apps) via the
    ``agentsysperf.report_generators`` entry point.

    Required class attributes:

    - ``name`` — short string id (e.g. ``"html"``).
    - ``output_formats`` — frozenset of file extensions this generator
      produces (e.g. ``frozenset(["html", "pdf"])``).
    """

    name: str
    output_formats: frozenset

    def generate_report(self, *, run_id: str, store: ResultStore, output_path: Path) -> Path:
        """Generate a report for the given run and return the output path.

        ``output_path`` is the target file path (with extension matching
        one of :attr:`output_formats`). The generator writes the report
        and returns the path (or raises on failure).
        """
        ...


# =============================================================================
# Plugin discovery — entry-point group names + helpers
# =============================================================================


# Pip packages register adapters via:
#
#   [project.entry-points."agentsysperf.benchmarks"]
#   terminal-bench = "my_pkg.adapter:TerminalBenchAdapter"
#
# and so on for the other groups. AgentSysPerf core ships built-in
# entries for its reference implementations.
ENTRY_POINT_BENCHMARKS = "agentsysperf.benchmarks"
ENTRY_POINT_MEASUREMENTS = "agentsysperf.measurements"
ENTRY_POINT_OPTIMIZATION_PROFILES = "agentsysperf.optimization_profiles"
ENTRY_POINT_HARDWARE_TELEMETRY = "agentsysperf.hardware_telemetry"
ENTRY_POINT_ANALYZERS = "agentsysperf.analyzers"
ENTRY_POINT_RESULT_STORES = "agentsysperf.result_stores"
ENTRY_POINT_REPORT_GENERATORS = "agentsysperf.report_generators"


def discover_benchmarks() -> Dict[str, BenchmarkAdapter]:
    """Return all installed benchmark adapters keyed by name."""
    return _discover(ENTRY_POINT_BENCHMARKS)


def discover_measurements() -> Dict[str, Measurement]:
    """Return all installed measurement plugins keyed by name."""
    return _discover(ENTRY_POINT_MEASUREMENTS)


def discover_optimization_profiles() -> Dict[str, OptimizationProfilePlugin]:
    """Return all installed optimization-profile plugins keyed by name."""
    return _discover(ENTRY_POINT_OPTIMIZATION_PROFILES)


def discover_hardware_telemetry() -> Dict[str, HardwareTelemetryPlugin]:
    """Return all installed hardware-telemetry plugins keyed by name."""
    return _discover(ENTRY_POINT_HARDWARE_TELEMETRY)


def discover_analyzers() -> Dict[str, Analyzer]:
    """Return all installed analyzer plugins keyed by name."""
    return _discover(ENTRY_POINT_ANALYZERS)


def discover_result_stores() -> Dict[str, "ResultStore"]:
    """Return all installed result-store plugins keyed by name."""
    return _discover(ENTRY_POINT_RESULT_STORES)


def discover_report_generators() -> Dict[str, "ReportGenerator"]:
    """Return all installed report-generator plugins keyed by name."""
    return _discover(ENTRY_POINT_REPORT_GENERATORS)


def _discover(group: str) -> Dict[str, Any]:
    """Lookup all entry points for ``group`` and instantiate them.

    Tolerates discovery failures: a plugin that errors on import is
    logged and skipped, not raised.  The user gets a working CLI even
    when a third-party plugin breaks.
    """
    import logging
    from importlib.metadata import entry_points

    log = logging.getLogger(__name__)
    out: Dict[str, Any] = {}
    try:
        eps = entry_points(group=group)
    except TypeError:  # pre-3.10 fallback
        eps = entry_points().get(group, [])  # type: ignore[union-attr]
    # Result stores need a target (a DB path / DSN), so they construct via a
    # zero-arg classmethod ``open()`` that resolves the canonical location.
    # Other plugin groups (analyzers, measurements, ...) construct with ``cls()``.
    # Without this, a store whose __init__ requires output_dir raised TypeError
    # here and was silently dropped from discovery.
    use_open = group == ENTRY_POINT_RESULT_STORES
    for ep in eps:
        try:
            cls = ep.load()
            factory = getattr(cls, "open", None)
            if use_open and callable(factory):
                out[ep.name] = factory()
            else:
                out[ep.name] = cls()
        except Exception:
            log.warning(
                "AgentSysPerf could not load %s plugin %r — skipping",
                group, ep.name, exc_info=True,
            )
    return out


# =============================================================================
# Sentinel: a no-op Measurement, useful as a base class or test double
# =============================================================================


class NullMeasurement:
    """Default Measurement that observes nothing.

    Use as a base class when you only need to override ``finalize_span``
    or ``stop``.  Also useful in tests as a placeholder.
    """

    name: str = "null"
    layer: str = "null"

    def start(self, *, run_id: str, output_dir: Path) -> None:
        return None

    def observe_span(self, *, span_id: str, kind: str, node_id: str) -> None:
        return None

    def finalize_span(self, span_id: str) -> Iterable[MeasurementRecord]:
        return ()

    def stop(self) -> Iterable[MeasurementRecord]:
        return ()


__all__ = [
    "TaskSpec",
    "TaskResult",
    "MeasurementRecord",
    "CounterReading",
    "AnalysisResult",
    "AnalysisContext",
    "BenchmarkAdapter",
    "AgentInvoker",
    "Measurement",
    "OptimizationProfilePlugin",
    "HardwareTelemetryPlugin",
    "Analyzer",
    "ResultStore",
    "ReportGenerator",
    "NullMeasurement",
    "ENTRY_POINT_BENCHMARKS",
    "ENTRY_POINT_MEASUREMENTS",
    "ENTRY_POINT_OPTIMIZATION_PROFILES",
    "ENTRY_POINT_HARDWARE_TELEMETRY",
    "ENTRY_POINT_ANALYZERS",
    "ENTRY_POINT_RESULT_STORES",
    "ENTRY_POINT_REPORT_GENERATORS",
    "discover_benchmarks",
    "discover_measurements",
    "discover_optimization_profiles",
    "discover_hardware_telemetry",
    "discover_analyzers",
    "discover_result_stores",
    "discover_report_generators",
]
