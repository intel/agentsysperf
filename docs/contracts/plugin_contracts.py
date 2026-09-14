#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""
=============================================================================
AgentSysPerf Plugin Contracts
=============================================================================

Concrete Python ABCs that pluggable entities must implement.
Mirrors AIPerf's plugin discovery via setuptools entry points.

Entity types covered:
  1. AgentFrameworkPlugin   — LangGraph / AutoGen / CrewAI / etc
  2. GatewayPlugin          — Envoy / Kong / LiteLLM / etc
  3. SchedulerPlugin        — K8s / node-local (which core)
  4. RoutingStrategyPlugin  — single_pool / least_loaded / kv_affinity /
                               wss / hybrid / ... (which pool — WSS is one)
  5. ClassifierPlugin       — rule-based / SLM / hybrid WSS classifier
  6. InferenceBackendPlugin — vLLM / TGI / Triton / Ollama
  7. ToolsRagPlugin         — Qdrant / Weaviate / external APIs
  8. MemoryStorePlugin      — Redis / in-memory / SQLite
  9. OptimizationProfilePlugin — base vs Xeon (AMX/oneDNN/NUMA/...) —
                               swept & engagement-verified, not assumed
 10. HardwareTelemetryPlugin — pcm / dcgmi / topdown / Prometheus
                               (also VERIFIES optimization engagement)
 11. QualityJudgePlugin     — LLM judge / regex / human queue
 12. WorkloadPlugin         — terminal_bench / synthetic / trace-replay / YAML

To implement a plugin:
  1. Subclass the relevant ABC
  2. Register via pyproject.toml entry_points:
       [project.entry-points."agentsysperf.agent_framework"]
       langgraph = "my_pkg.agentsysperf_langgraph:LangGraphAdapter"
  3. AgentSysPerf discovers and instantiates by name at runtime

NOTE: This file declares CONTRACTS only. Implementations live in separate
packages so the core framework has zero dependencies on user frameworks.
=============================================================================
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any, AsyncIterator, Iterator, Literal, Optional, Protocol
)
from datetime import datetime


# =============================================================================
# CORE DATA MODELS
# =============================================================================

@dataclass(frozen=True)
class AgenticTask:
    """One unit of benchmark work — a complete agentic task."""

    task_id: str
    task_type: Literal[
        "legal_review", "code_agent", "rag_qa",
        "customer_support", "research", "custom"
    ]

    # Input payload
    user_prompt: str
    context: dict[str, Any] = field(default_factory=dict)

    # Expected behavior (drives quality judging — optional)
    expected_intent: Optional[str] = None
    expected_tools_called: Optional[list[str]] = None
    expected_answer_contains: Optional[list[str]] = None
    ground_truth_step_wss: Optional[dict[str, str]] = None  # step_name → WSS class

    # SLOs
    deadline_s: float = 30.0
    cost_budget_usd: Optional[float] = None

    # Determinism
    seed: int = 0
    framework_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepTrace:
    """One step inside an agent task."""

    step_id: str
    step_name: str
    started_at: float
    finished_at: float

    # Component-level timing breakdown
    framework_overhead_ms: float = 0.0
    gateway_latency_ms: float = 0.0
    backend_latency_ms: float = 0.0
    tool_latency_ms: float = 0.0

    # Routing observations
    wss_class: Optional[str] = None       # "tiny" | "small" | "medium" | "large"
    routed_to_pool: Optional[str] = None
    routing_decision_ms: float = 0.0

    # Tokens & cost
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0

    # Cache observations
    kv_cache_hit: bool = False
    semantic_cache_hit: bool = False

    # Free-form for plugin-specific data
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Final outcome of a benchmark task."""

    task_id: str
    success: bool
    final_answer: str
    error: Optional[str] = None

    # Per-step trace
    steps: list[StepTrace] = field(default_factory=list)

    # Aggregates (filled by framework; plugins don't need to compute)
    total_latency_s: float = 0.0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_cost_usd: float = 0.0
    tool_calls_made: list[str] = field(default_factory=list)
    pools_used: dict[str, int] = field(default_factory=dict)
    kv_cache_hits: int = 0
    kv_cache_misses: int = 0


@dataclass
class QualityScore:
    """Output of the quality judge."""

    task_id: str
    success: bool                          # binary pass/fail
    success_score: float                   # 0.0–1.0
    intent_correct: Optional[bool] = None
    answer_contains_required: Optional[bool] = None
    tool_precision: Optional[float] = None
    tool_recall: Optional[float] = None
    judge_reasoning: Optional[str] = None  # for human review
    judge_name: str = "unknown"


@dataclass
class RoutingStats:
    """Per-window routing observations from the gateway."""

    window_start_ts: float
    window_end_ts: float
    total_requests: int
    routes_per_pool: dict[str, int]
    fallback_count: int = 0
    cold_route_count: int = 0
    rejected_count: int = 0


@dataclass
class PoolMetrics:
    """Per-pool inference backend metrics."""

    pool_name: str
    timestamp: float
    in_flight_requests: int
    kv_cache_hit_rate: float
    queue_depth: int
    gpu_utilization_pct: Optional[float] = None
    cpu_utilization_pct: Optional[float] = None
    memory_used_mb: float = 0.0


# =============================================================================
# 1. AGENT FRAMEWORK PLUGIN
# =============================================================================

class AgentFrameworkPlugin(ABC):
    """Adapter for an agent runtime (LangGraph, AutoGen, CrewAI, ...).

    The adapter submits a task to the framework, lets it run end-to-end,
    and returns the trace. The adapter is responsible for translating the
    AgenticTask into whatever shape the framework expects, and for
    instrumenting each step to emit StepTrace events.
    """

    @property
    @abstractmethod
    def framework_name(self) -> str:
        """Stable identifier, e.g. 'langgraph', 'autogen', 'crewai'."""

    @property
    @abstractmethod
    def framework_version(self) -> str:
        """Version string for reproducibility."""

    @abstractmethod
    async def submit_task(self, task: AgenticTask) -> TaskResult:
        """Run the task to completion (or deadline). Return full result."""

    @abstractmethod
    async def health(self) -> bool:
        """Liveness check. False = adapter unhealthy, abort benchmark."""

    async def warmup(self, sample_tasks: list[AgenticTask]) -> None:
        """Optional pre-warm with sample tasks. Default no-op."""
        return None

    async def shutdown(self) -> None:
        """Cleanup at end of benchmark. Default no-op."""
        return None


# =============================================================================
# 2. GATEWAY PLUGIN
# =============================================================================

class GatewayPlugin(ABC):
    """Adapter for the AI gateway (Envoy / Kong / LiteLLM / vLLM router).

    The framework doesn't drive the gateway directly — the agent framework
    does. But the GatewayPlugin lets AgentSysPerf inspect gateway state,
    routing decisions, and rate-limit headroom while the benchmark runs.
    """

    @property
    @abstractmethod
    def gateway_name(self) -> str:
        ...

    @abstractmethod
    async def get_routing_stats(self, since_ts: float) -> RoutingStats:
        """Snapshot of routing decisions since `since_ts`."""

    @abstractmethod
    async def health(self) -> bool:
        ...

    async def reset_metrics(self) -> None:
        """Optional: reset counters between benchmark phases."""
        return None


# =============================================================================
# 3. SCHEDULER PLUGIN
# =============================================================================

class SchedulerPlugin(ABC):
    """Adapter for the work scheduler (K8s / node-local).

    NOTE: scope boundary. The scheduler decides WHERE ON CORES a unit of work
    runs once it has arrived at a pool. It does NOT decide which pool. Pool
    selection is the RoutingStrategyPlugin's job (contract 4 below). Keeping
    these separate is deliberate: production systems separate "which pool"
    (router/gateway) from "which core" (scheduler), and the benchmark must be
    able to vary one without perturbing the other.
    """

    @property
    @abstractmethod
    def scheduler_name(self) -> str:
        ...

    @abstractmethod
    async def get_pool_loads(self) -> dict[str, float]:
        """pool_name → utilization 0.0–1.0."""

    @abstractmethod
    async def get_pending_queue_depths(self) -> dict[str, int]:
        """pool_name → pending work units."""


# =============================================================================
# 4. ROUTING STRATEGY PLUGIN
# =============================================================================
# WSS routing is NOT special-cased anywhere in AgentSysPerf. It is ONE registered
# routing strategy among many. The benchmark's job is to MEASURE whether WSS
# routing beats the alternatives — not to assume it does. Treating routing
# strategy as a swept, pluggable variable is what makes that measurement
# defensible.
#
# A routing strategy is a policy: (request features, system state) -> pool.
# Strategies differ by WHICH inputs they consume and HOW. Reference strategies
# shipped with AgentSysPerf:
#
#   single_pool   — one pool, no decision (null baseline)
#   round_robin   — rotation counter; no request/state awareness
#   random        — RNG pick
#   least_loaded  — fewest in-flight requests (load-aware)
#   power_of_two  — 2 random samples, pick lighter (near-optimal LB)
#   kv_affinity   — prefix-hash → warm-cache pool (strong production baseline)
#   model_based   — route by which model the step needs
#   slo_aware     — route by remaining deadline / SLA tier
#   wss           — payload size → cache tier → pool (the HYPOTHESIS UNDER TEST)
#   semantic_slm  — tiny model predicts route from prompt semantics
#   hybrid        — composition (e.g. model filter → kv_affinity → least_loaded)
#
# A strategy MUST declare the signals it requires. If the SUT cannot supply a
# required signal, AgentSysPerf fails loudly rather than silently degrading the
# strategy to random (which would corrupt the comparison).
# =============================================================================

@dataclass
class RoutingRequestFeatures:
    """What the router may inspect about the request."""
    task_type: str
    step_name: str
    payload_bytes: int
    prompt_prefix_hash: Optional[str] = None   # for kv_affinity
    model_required: Optional[str] = None       # for model_based
    deadline_s: Optional[float] = None         # for slo_aware
    sla_tier: Optional[str] = None             # for slo_aware
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingSystemState:
    """What the router may inspect about the system at decision time.

    Fields are Optional: a strategy that needs one not provided by the SUT
    must surface that via `required_signals` so AgentSysPerf can refuse to run
    rather than compare a crippled strategy.
    """
    pool_loads: Optional[dict[str, float]] = None        # pool → utilization
    pool_queue_depths: Optional[dict[str, int]] = None   # pool → pending
    pool_kv_cache_keys: Optional[dict[str, set]] = None  # pool → cached prefixes
    pool_models: Optional[dict[str, set]] = None         # pool → models loaded
    healthy_pools: Optional[set] = None


@dataclass
class RoutingDecision:
    target_pool: str
    decision_cost_ms: float          # charged back into StepTrace
    strategy_name: str
    signals_used: list[str] = field(default_factory=list)
    fell_back: bool = False          # required signal missing → degraded path


class RoutingStrategyPlugin(ABC):
    """A pool-selection policy. WSS is one implementation, not the contract."""

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Stable id: 'wss', 'least_loaded', 'kv_affinity', 'hybrid', ..."""

    @property
    @abstractmethod
    def required_signals(self) -> set:
        """Subset of RoutingSystemState / RoutingRequestFeatures field names
        this strategy cannot function without. AgentSysPerf validates these are
        available BEFORE the run and aborts with a clear error if not.
        Example: least_loaded -> {'pool_loads'};
                 kv_affinity  -> {'prompt_prefix_hash', 'pool_kv_cache_keys'};
                 round_robin  -> set()  (needs nothing)."""

    @property
    @abstractmethod
    def is_content_aware(self) -> bool:
        """True if the decision depends on request content (wss, kv_affinity,
        model_based, semantic_slm, slo_aware). Used by the reporter to group
        results and to enforce that decision cost is charged."""

    @abstractmethod
    async def decide(
        self,
        features: RoutingRequestFeatures,
        state: RoutingSystemState,
        available_pools: list[str],
    ) -> RoutingDecision:
        """Return the target pool plus the cost of deciding. For content-aware
        strategies the decision_cost_ms MUST be > 0 and is charged into the
        step's routing_decision_ms — otherwise WSS / semantic_slm would look
        artificially cheap versus round_robin."""

    async def reset(self) -> None:
        """Reset internal state (e.g. round-robin counter) between phases."""
        return None


# =============================================================================
# 5. CLASSIFIER PLUGIN
# =============================================================================

class ClassifierPlugin(ABC):
    """Adapter for the WSS classifier (rule-based, SLM, hybrid).

    AgentSysPerf can call this directly to evaluate classifier accuracy
    against ground-truth labels in the workload.
    """

    @property
    @abstractmethod
    def classifier_name(self) -> str:
        ...

    @abstractmethod
    async def classify(self, prompt: str, task_type: str) -> str:
        """Return WSS class: 'tiny' | 'small' | 'medium' | 'large'."""

    @abstractmethod
    async def classify_batch(
        self, prompts: list[str], task_types: list[str]
    ) -> list[str]:
        """Batch version for evaluation runs. Default = serial classify."""


# =============================================================================
# 6. INFERENCE BACKEND PLUGIN
# =============================================================================

class InferenceBackendPlugin(ABC):
    """Adapter for inference workers (vLLM, TGI, Triton, llama.cpp, Ollama).

    AgentSysPerf doesn't drive inference directly (the agent framework does).
    But this plugin lets us inspect per-pool metrics during a benchmark.
    """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        ...

    @abstractmethod
    async def get_pool_metrics(self, pool_name: str) -> PoolMetrics:
        ...

    @abstractmethod
    async def list_pools(self) -> list[str]:
        ...

    async def warmup(self, prompts: list[str]) -> None:
        """Optional pre-warm of KV caches. Default no-op."""
        return None


# =============================================================================
# 7. TOOLS / RAG PLUGIN
# =============================================================================

class ToolsRagPlugin(ABC):
    """Adapter for tool / RAG backends (Qdrant, Weaviate, external APIs).

    AgentSysPerf observes tool-call latency per call type for attribution.
    """

    @property
    @abstractmethod
    def tools_name(self) -> str:
        ...

    @abstractmethod
    async def get_tool_call_stats(self, since_ts: float) -> dict[str, dict[str, float]]:
        """tool_name → {count, p50_ms, p99_ms, error_rate}."""


# =============================================================================
# 8. MEMORY STORE PLUGIN
# =============================================================================

class MemoryStorePlugin(ABC):
    """Adapter for session / KV / semantic memory."""

    @property
    @abstractmethod
    def memory_name(self) -> str:
        ...

    @abstractmethod
    async def get_cache_stats(self) -> dict[str, float]:
        """Return: {hit_rate, miss_rate, eviction_rate, size_mb}."""


# =============================================================================
# 9. OPTIMIZATION PROFILE PLUGIN
# =============================================================================
# "base vs Xeon-optimized" is NOT a feature toggle. It is a swept variable,
# exactly like routing_strategy. An OptimizationProfile owns a declarative
# set of build/runtime/OS optimization axes and PROJECTS them into the
# inference-backend and scheduler plugins. The benchmark measures what the
# Xeon tuning actually buys for agentic workloads — it does not assume it.
#
# Two integrity rules, mirroring the routing-study safeguards:
#
#  1. The `base` arm must be a REASONABLE PORTABLE build, never a strawman.
#     Every arm's full axis settings are published with results so a crippled
#     baseline cannot inflate a speedup multiple unnoticed.
#
#  2. A claimed optimization that did not actually engage INVALIDATES the arm.
#     "Built with AMX" means nothing if the kernel silently fell back to
#     AVX-512. Engagement is VERIFIED via HardwareTelemetryPlugin counters
#     (contract 10) and a non-engaged arm aborts — it is never reported as a
#     valid data point. This is the same "verify, don't trust the label"
#     discipline used for routing required_signals.
# =============================================================================

# Optimization axes. Each is independently togglable so a gain can be
# ATTRIBUTED (was it AMX, NUMA pinning, or quantization?) rather than hidden
# inside one opaque "optimized" bundle.
OPTIMIZATION_AXES = (
    "isa",            # generic_avx2 | avx512 | amx_tdpbssd
    "math_library",   # reference | openblas | onednn | onemkl
    "runtime",        # vanilla | ipex | openvino
    "quantization",   # fp32 | bf16 | int8_vnni | int4_amx
    "numa",           # unpinned | numa_local
    "memory_pages",   # 4k | hugepages_2m | hugepages_1g
    "core_isolation", # shared | isolcpus_affinity
    "accelerator",    # none | qat | dsa | iaa  (may be a set)
)


@dataclass
class OptimizationProfile:
    """Declarative optimization state. `name` labels the study arm
    (e.g. 'base', 'amx_only', 'full_xeon'). `axes` maps each axis in
    OPTIMIZATION_AXES to its chosen setting. `verify_counters` names the
    HardwareTelemetry counter(s) that must show non-trivial activity for
    each enabled optimization, plus the minimum threshold that counts as
    'engaged'. Missing/!engaged → arm aborts."""
    name: str
    axes: dict[str, Any]                       # axis -> setting
    verify_counters: dict[str, float] = field(default_factory=dict)
    # e.g. {"amx_active_cycle_ratio": 0.05, "hugepage_fault_ratio": 0.5,
    #       "numa_remote_access_ratio_max": 0.15}
    notes: str = ""


@dataclass
class ProfileProjection:
    """What a profile pushes into the other plugins. Returned by project()
    so the harness can apply backend settings and scheduler settings without
    the profile plugin needing to know either plugin's internals."""
    backend_config: dict[str, Any] = field(default_factory=dict)
    scheduler_config: dict[str, Any] = field(default_factory=dict)
    required_telemetry_counters: list[str] = field(default_factory=list)


@dataclass
class ProfileEngagementReport:
    """Result of verifying an applied profile actually took effect."""
    profile_name: str
    engaged: bool
    per_axis: dict[str, bool]                  # axis -> did it engage?
    measured_counters: dict[str, float]
    failures: list[str] = field(default_factory=list)


class OptimizationProfilePlugin(ABC):
    """Owns an OptimizationProfile and projects it into backend + scheduler.

    The harness lifecycle is:
        profile = plugin.get_profile()
        proj    = plugin.project(profile)
        # harness applies proj.backend_config to InferenceBackendPlugin
        # harness applies proj.scheduler_config to SchedulerPlugin
        # ... run warmup ...
        report  = plugin.verify_engaged(profile, telemetry)   # via contract 10
        if not report.engaged: ABORT THE ARM  (do not report it)
    """

    @property
    @abstractmethod
    def profile_name(self) -> str:
        """Stable arm id: 'base', 'amx_only', 'amx_onednn',
        'amx_numa_hugepages', 'full_xeon', ..."""

    @abstractmethod
    def get_profile(self) -> OptimizationProfile:
        """Return the declarative profile this plugin represents."""

    @abstractmethod
    def project(self, profile: OptimizationProfile) -> ProfileProjection:
        """Translate the profile into concrete backend + scheduler config.
        This is the projection layer that keeps the profile decoupled from
        any specific backend/scheduler implementation."""

    @abstractmethod
    async def verify_engaged(
        self,
        profile: OptimizationProfile,
        telemetry: "HardwareTelemetryPlugin",
        node_id: str,
    ) -> ProfileEngagementReport:
        """Read hardware counters and confirm each enabled optimization
        actually took effect. MUST return engaged=False (with failures
        populated) if any claimed optimization shows ~zero activity. The
        harness aborts non-engaged arms — they are never valid data points."""

    @property
    @abstractmethod
    def is_baseline(self) -> bool:
        """True only for the portable reference arm. The reporter flags the
        study invalid if the baseline arm is itself heavily optimized, so a
        strawman base cannot inflate the speedup."""


# =============================================================================
# 10. HARDWARE TELEMETRY PLUGIN
# =============================================================================

class HardwareTelemetryPlugin(ABC):
    """Adapter for hardware perf counters (pcm, dcgmi, rocm-smi, perf).

    Beyond raw metrics, this contract is the VERIFIER for optimization
    profiles (contract 9). `read_engagement_counters` must expose the
    specific counters that prove a Xeon optimization engaged — AMX-active
    cycle ratio, hugepage fault ratio, NUMA-remote access ratio, accelerator
    queue depth — so OptimizationProfilePlugin.verify_engaged can abort arms
    where the claimed optimization silently fell back."""

    @property
    @abstractmethod
    def hardware_name(self) -> str:
        ...

    @abstractmethod
    async def get_node_metrics(self, node_id: str) -> dict[str, float]:
        """Per-node telemetry — CPU%, GPU%, mem BW, AMX cycles, IO LLC hits, etc."""

    @abstractmethod
    async def list_nodes(self) -> list[str]:
        ...

    @abstractmethod
    async def read_engagement_counters(
        self, node_id: str, counter_names: list[str]
    ) -> dict[str, float]:
        """Return the named low-level counters used to PROVE an optimization
        engaged (not the config label). Examples:
          amx_active_cycle_ratio   — AMX tile cycles / total busy cycles
          hugepage_fault_ratio     — 2M/1G faults / total page faults
          numa_remote_access_ratio — remote / total memory accesses
          accel_queue_depth_qat    — QAT submission depth
        Raise if a requested counter is unavailable on this hardware — the
        profile arm must then abort rather than be trusted on its label."""


# =============================================================================
# 11. QUALITY JUDGE PLUGIN
# =============================================================================

class QualityJudgePlugin(ABC):
    """Evaluates whether a task was completed correctly.

    Implementations:
      - LLM-as-judge (Claude, GPT-4o)
      - Regex / deterministic check
      - Human review queue (returns pending until human votes)
      - Composite (majority of N judges)
    """

    @property
    @abstractmethod
    def judge_name(self) -> str:
        ...

    @abstractmethod
    async def judge(
        self, task: AgenticTask, result: TaskResult
    ) -> QualityScore:
        ...

    async def batch_judge(
        self, pairs: list[tuple[AgenticTask, TaskResult]]
    ) -> list[QualityScore]:
        """Default = serial. Override for batched LLM-judge cost savings."""
        return [await self.judge(t, r) for t, r in pairs]


# =============================================================================
# 12. WORKLOAD PLUGIN
# =============================================================================

class WorkloadPlugin(ABC):
    """Generates agentic tasks. Synthetic, trace-replay, or YAML-driven."""

    @property
    @abstractmethod
    def workload_name(self) -> str:
        ...

    @abstractmethod
    def generate(
        self, count: int, seed: int, config: dict[str, Any]
    ) -> Iterator[AgenticTask]:
        """Yield N tasks deterministically given (count, seed, config)."""

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Return metadata: expected steps, SLO defaults, parameter ranges."""


# =============================================================================
# PLUGIN REGISTRY
# =============================================================================

class PluginRegistry:
    """Discovers plugins via setuptools entry_points at startup.

    Entry-point groups expected:
      agentsysperf.agent_framework
      agentsysperf.gateway
      agentsysperf.scheduler
      agentsysperf.routing_strategy
      agentsysperf.classifier
      agentsysperf.inference_backend
      agentsysperf.tools_rag
      agentsysperf.memory_store
      agentsysperf.optimization_profile
      agentsysperf.hardware_telemetry
      agentsysperf.quality_judge
      agentsysperf.workload
    """

    def __init__(self):
        self._plugins: dict[str, dict[str, type]] = {}

    def discover(self) -> None:
        """Scan installed packages for entry-point registrations."""
        from importlib.metadata import entry_points

        groups = [
            "agent_framework", "gateway", "scheduler", "routing_strategy",
            "classifier", "inference_backend", "tools_rag", "memory_store",
            "optimization_profile", "hardware_telemetry", "quality_judge",
            "workload",
        ]
        for group in groups:
            self._plugins[group] = {}
            for ep in entry_points(group=f"agentsysperf.{group}"):
                self._plugins[group][ep.name] = ep.load()

    def get(self, group: str, name: str) -> type:
        """Return plugin class. Raises if not found."""
        if group not in self._plugins:
            raise KeyError(f"Unknown plugin group: {group}")
        if name not in self._plugins[group]:
            available = list(self._plugins[group].keys())
            raise KeyError(
                f"Unknown plugin '{name}' in group '{group}'. "
                f"Available: {available}"
            )
        return self._plugins[group][name]

    def list_group(self, group: str) -> list[str]:
        return list(self._plugins.get(group, {}).keys())
