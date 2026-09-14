# AgentSysPerf — Design Proposal

**Status:** Draft proposal, not yet implemented
**Inspired by:** [AIPerf](https://github.com/ai-dynamo/aiperf) (single-endpoint LLM benchmarking)
**Goal:** Open-source benchmark suite for **end-to-end agentic AI stacks** with pluggable entities

---

## 1. Motivation

### What AIPerf Measures

AIPerf is excellent at one job: load-testing a single inference endpoint. It reports TTFT, ITL, throughput, output length distributions. Plug in vLLM, TGI, Triton, or Ollama — get apples-to-apples token-level numbers.

### What AIPerf Cannot Measure

The agentic AI stack has at least 6 components beyond the model server:

1. **Agent framework** (LangGraph, AutoGen, CrewAI) — orchestrates multi-step plans
2. **Gateway / router** (Envoy, Kong, LiteLLM) — Layer 2 routing decisions
3. **Scheduler** (K8s, node-local) — work distribution
4. **Classifier** (rule-based, SLM-based) — working-set-size analysis
5. **Tools / RAG** (vector DB, external APIs) — augment the model
6. **Hardware** (Xeon AP/SP, GPU, Arm) — substrate

A "fast model" doesn't make a fast agent. A 50ms TTFT means nothing if the agent spends 800ms in the gateway and 1.2s on a slow RAG lookup. **The bottleneck is almost never the model.**

### What Production Teams Actually Need

Real questions teams ask that AIPerf cannot answer:

- "How does swapping LangGraph for AutoGen affect end-to-end task latency?"
- "Does our node-local scheduler (the design from earlier) actually beat a centralized gateway at p99?"
- "What's the cost-per-task on Xeon-AP vs. GPU for this workflow?"
- "How does our routing accuracy change if we replace the rule-based classifier with a 1B SLM?"
- "What's the KV cache hit rate across multi-turn conversations in this stack?"
- "Where in the agent does p99 latency actually live — model, gateway, or RAG?"

These are **stack-level** questions. They require a benchmark that treats the agent stack as a system, not a single endpoint.

---

## 2. Goals & Non-Goals

### Goals

✅ **Measure end-to-end agentic task performance**, not just token throughput
✅ **Pluggable everything** — swap agent framework, gateway, scheduler, backend, hardware
✅ **Standardized workloads** — common task suites for apples-to-apples comparison
✅ **Production-realistic** — multi-turn, multi-tool, multi-step task graphs
✅ **Reproducible** — seedable, container-based reference deployments
✅ **Observability-first** — consume OTEL traces from the stack, don't replace them
✅ **Cost-aware** — report $/task, not just latency
✅ **Quality-aware** — task success rate, not just liveness

### Non-Goals

❌ Replacing AIPerf for single-endpoint benchmarks (use AIPerf for that — and we depend on it for the inference layer)
❌ Being a production agent framework (AgentSysPerf is a measurement tool, not a runtime)
❌ Evaluating *model quality* in a general sense — we measure **task-completion quality** within a specific agentic workflow, not general capability (use lm-eval-harness, HELM, MMLU for capability eval)
❌ Mandating one specific framework — every entity must be pluggable

---

## 3. Architecture

### Three Planes (mirroring AIPerf's design)

| Plane | Components | Responsibility |
|-------|-----------|----------------|
| **Control** | Orchestrator, Plugin Loader, Metric Aggregator, Reporter | Spawn workers, manage config, collect & report |
| **Data** | Workload Generator, Task Driver Pool, Trace Collector | Generate tasks, drive them through the SUT, collect traces |
| **System Under Test (SUT)** | Plugin adapters → user's agent stack | Whatever the user wants benchmarked |

### Why Three Planes

AIPerf already proved this works. The separation gives us:
- Control plane scales independently from data plane (one orchestrator, many drivers)
- SUT is fully decoupled (different message bus, different process, different host)
- Plugins can be added without touching framework code

### Process Topology

```
┌───────────────────────────────────────────────────────────┐
│ AgentSysPerf Orchestrator (Python, asyncio)                  │
│   - Loads plugins by entry-point                          │
│   - Spawns N task driver processes                        │
│   - ZMQ pub/sub for metrics aggregation                   │
└───────────────────────────────────────────────────────────┘
                       │ ZMQ INPROC + IPC
                       ↓
        ┌──────────────┴──────────────┐
        ↓                             ↓
   ┌──────────┐                  ┌──────────┐
   │ Driver 0 │  ...             │ Driver N │   (each = process)
   └──────────┘                  └──────────┘
        │                             │
        │ Calls plugin adapter        │
        ↓                             ↓
   ┌────────────────────────────────────────┐
   │ User's agent stack (over HTTP/gRPC)    │
   │ - LangGraph endpoint                   │
   │ - Envoy gateway                        │
   │ - vLLM workers                         │
   │ - Vector DB                            │
   └────────────────────────────────────────┘
        │
        │ OTEL spans + structured logs
        ↓
   ┌────────────────────────────────────────┐
   │ Trace Collector (OTEL exporter)        │
   │ Joins driver-side timings with         │
   │ server-side spans → per-step attribute │
   └────────────────────────────────────────┘
```

---

## 4. Pluggable Entity Contracts

Each pluggable entity is a Python ABC. Implementations register via entry points in `pyproject.toml` (same pattern AIPerf uses).

```python
# src/plugins/agent_framework.py

class AgentFrameworkPlugin(ABC):
    """Adapter for an agent runtime. Submits one task; collects the trace."""

    @abstractmethod
    async def submit_task(self, task: AgenticTask, deadline_s: float) -> TaskResult:
        """Run the task end-to-end. Return final answer + per-step trace."""

    @abstractmethod
    async def health(self) -> bool:
        """Liveness check."""

    @property
    @abstractmethod
    def framework_name(self) -> str:
        ...
```

```python
# src/plugins/gateway.py

class GatewayPlugin(ABC):
    """Adapter for the AI gateway under test."""

    @abstractmethod
    async def get_routing_stats(self, since_ts: float) -> RoutingStats:
        """Return per-pool routed counts and decisions."""

    @abstractmethod
    async def health(self) -> bool:
        ...
```

```python
# src/plugins/inference_backend.py

class InferenceBackendPlugin(ABC):
    """Adapter for inference workers (vLLM, TGI, Triton, etc.)."""

    @abstractmethod
    async def get_pool_metrics(self, pool_name: str) -> PoolMetrics:
        """Per-pool KV hit rate, in-flight count, GPU utilization."""

    @abstractmethod
    async def warmup(self, prompts: list[str]) -> None:
        """Pre-warm caches before benchmark starts."""
```

```python
# src/plugins/workload.py

class WorkloadPlugin(ABC):
    """Generates agentic tasks. Synthetic, trace-replay, or YAML."""

    @abstractmethod
    def generate(self, count: int, seed: int) -> Iterator[AgenticTask]:
        """Yield N tasks deterministically."""
```

```python
# src/plugins/quality_judge.py

class QualityJudgePlugin(ABC):
    """Evaluates whether a task was completed correctly."""

    @abstractmethod
    async def judge(
        self, task: AgenticTask, result: TaskResult
    ) -> QualityScore:
        """Return success/fail + reasoning. May use LLM-as-judge,
        regex, deterministic check, or human eval queue."""
```

The full contract set is **twelve** plugins (see `plugin_contracts.py` for the authoritative ABCs):

| # | Contract | Decides / supplies | Reference impls |
|---|----------|--------------------|-----------------|
| 1 | `AgentFrameworkPlugin` | runs the multi-step task | LangGraph, AutoGen, CrewAI |
| 2 | `GatewayPlugin` | request ingress, routing stats | Envoy, LiteLLM, Kong, passthrough |
| 3 | `SchedulerPlugin` | which **core** work runs on | K8s, node-local |
| 4 | `RoutingStrategyPlugin` | which **pool** a step goes to — WSS is one of eleven strategies, *measured not assumed* | `routing_strategies.py` (single_pool, round_robin, random, least_loaded, power_of_two, kv_affinity, model_based, slo_aware, wss, semantic_slm, hybrid) |
| 5 | `ClassifierPlugin` | WSS class of a step | rule-based, 1B SLM, hybrid |
| 6 | `InferenceBackendPlugin` | token generation | vLLM, TGI, Triton, Ollama |
| 7 | `ToolsRagPlugin` | retrieval / external tools | Qdrant, Weaviate, Chroma |
| 8 | `MemoryStorePlugin` | session / KV / semantic cache | Redis, in-memory, SQLite |
| 9 | `OptimizationProfilePlugin` | base vs Xeon build/runtime/OS axes — *swept and engagement-verified, never assumed* | `optimization_profiles.py` (base, amx_only, amx_onednn, amx_numa_hugepages, openvino_xeon, full_xeon) |
| 10 | `HardwareTelemetryPlugin` | perf counters **and** optimization-engagement verification | pcm, perf, dcgmi, Prometheus |
| 11 | `QualityJudgePlugin` | task success scoring | LLM-judge, regex, deterministic, human |
| 12 | `WorkloadPlugin` | generates / supplies tasks | terminal_bench, synthetic, trace-replay, YAML |

Contracts 4 and 9 are the **swept variables** that make WSS routing and Xeon optimization measured hypotheses rather than built-in assumptions (see §8 and §9). Contract 10 doubles as the verifier for contract 9 and the source for the hardware fingerprint catalog (see §6).

---

## 5. Core Data Models

### AgenticTask — a unit of benchmark work

```python
@dataclass
class AgenticTask:
    task_id: str
    task_type: Literal[
        "legal_review",     # summarize + find similar + revise
        "code_agent",       # plan + edit + test + iterate
        "rag_qa",           # retrieve + answer
        "customer_support", # intent + tool + reply
        "research",         # search + synthesize + cite
        "custom",
    ]

    # Inputs
    user_prompt: str
    context: dict[str, Any]      # documents, history, etc.

    # Expected behavior (for quality judging)
    expected_intent: str | None
    expected_tools_called: list[str] | None
    expected_answer_contains: list[str] | None

    # SLOs
    deadline_s: float
    cost_budget_usd: float | None

    # Determinism
    seed: int
```

### TaskResult — what comes back

```python
@dataclass
class TaskResult:
    task_id: str
    success: bool
    final_answer: str

    # Per-step trace (filled by the agent adapter)
    steps: list[StepTrace]

    # Aggregated metrics
    total_latency_s: float
    total_tokens_in: int
    total_tokens_out: int
    total_cost_usd: float
    tool_calls_made: list[str]

    # Routing observations
    pools_used: dict[str, int]   # pool_name → step_count
    kv_cache_hits: int
    kv_cache_misses: int
```

### StepTrace — one step inside the agent

```python
@dataclass
class StepTrace:
    step_id: str
    step_name: str               # "classify", "summarize", "rag_retrieve", ...
    started_at: float
    finished_at: float

    # Component breakdown
    framework_overhead_ms: float
    gateway_latency_ms: float
    backend_latency_ms: float
    tool_latency_ms: float

    # Working set classification
    wss_class: str               # "tiny" | "small" | "medium" | "large"
    routed_to_pool: str

    # Token economics
    tokens_in: int
    tokens_out: int
    cost_usd: float

    # Cache observations
    kv_cache_hit: bool
    semantic_cache_hit: bool
```

---

## 6. Metric Taxonomy

Four families. Each metric has `avg / min / max / p50 / p90 / p99 / std` (same as AIPerf's table style).

### 6.1 Time Metrics

| Metric | Definition | Why It Matters |
|--------|------------|----------------|
| `task_latency_e2e_ms` | Time from task submit to final answer | The user's experience |
| `step_latency_ms{step_name}` | Per-step duration | Pinpoint slow steps |
| `framework_overhead_ms` | Time inside the agent framework, not in backends | LangGraph vs CrewAI cost |
| `gateway_latency_ms` | Time in gateway routing | Layer 2 efficiency |
| `routing_decision_ms` | Time spent classifying + dispatching | WSS classifier cost |
| `first_step_ttft_ms` | Time to first step starting | Cold-start sensitivity |
| `tool_call_latency_ms{tool}` | Per-tool latency | RAG vs API vs function overhead |
| `kv_cache_hit_latency_ms` | Latency when KV cache hits | Reuse benefit |
| `kv_cache_miss_latency_ms` | Latency when KV cache misses | Cold path cost |

### 6.2 Quality Metrics

| Metric | Definition |
|--------|------------|
| `task_success_rate` | Tasks meeting the quality bar / total tasks |
| `intent_classification_accuracy` | Did the agent identify intent correctly? |
| `tool_call_precision` | Tools called that should have been / tools called |
| `tool_call_recall` | Tools called that should have been / tools that should have been called |
| `routing_accuracy` | Step routed to correct pool / total steps (against ground-truth WSS labels) |
| `hallucination_rate` | Outputs with detected fabrications (optional LLM judge) |
| `cite_coverage` | Citations linked to retrieved docs / claims requiring citation |

### 6.3 Resource & Cost Metrics

| Metric | Definition |
|--------|------------|
| `tokens_in_per_task` | Total prompt tokens across all steps |
| `tokens_out_per_task` | Total completion tokens across all steps |
| `cost_per_task_usd` | Tokens × price + GPU-time + CPU-time + tool-call fees |
| `cpu_core_seconds_per_task` | Aggregate CPU time |
| `gpu_seconds_per_task` | Aggregate GPU time |
| `memory_peak_mb_per_task` | Peak working set across all components |

### 6.4 Routing & Scheduling Metrics

| Metric | Definition |
|--------|------------|
| `wss_classification_accuracy` | Predicted class matches ground-truth bucket |
| `pool_utilization{pool}` | Per-pool busy time fraction |
| `cross_pool_handoffs_per_task` | Steps routed to a different pool than previous step |
| `cold_route_rate` | Routing decisions that hit a cold backend (no KV cache, no warm session) |
| `fallback_rate` | Times the gateway fell back to a non-preferred pool |

### 6.5 Hardware Fingerprint (Xeon perf characterization)

Beyond the aggregate metrics above, AgentSysPerf can attach a **per-stage hardware fingerprint** sourced through `HardwareTelemetryPlugin` (contract 10) — TMA top-down breakdown, IPC, dominant cache tier, per-socket memory bandwidth, AMX-active ratio, package power. This is what lets a stage be labelled "memory-bound, AMX-accelerated" vs "core-efficient, L2-resident", and is how the WSS routing rule table is *verified* per stage rather than assumed.

The full counter list, granularity rules (what is genuinely per-core vs per-socket), Intel tool mapping (pcm / perf / toplev / emon), and the cross-cutting caveats (sampling interval vs stage duration, per-core attribution of shared resources being a heuristic, counter multiplexing, observer effect) are in **`hw_metrics_catalog.md`** in this package. Per-stage attribution is only meaningful when core isolation and WSS pool-pinning are active — i.e. the fingerprint is the *payoff* of the §8/§9 work, not a standalone feature.

---

## 7. Standard Workload Suites

To enable apples-to-apples comparison, ship reference workloads. Each is a YAML spec + Python generator.

### 7.1 `legal_review` (medium WSS, deep agent)

Multi-step contract review (the example from our earlier work):

```yaml
workload: legal_review
task_count: 100
seed: 42
parameters:
  contract_size_tokens: [500, 2000, 8000]   # mix of sizes
  similar_clause_db_size: 10000
  num_revisions_per_contract: [1, 3, 5]
expected_steps:
  - classify
  - plan
  - summarize
  - rag_retrieve
  - rerank
  - propose_revisions
  - format
slo:
  p99_latency_s: 3.0
  task_success_rate: 0.95
```

### 7.2 `rag_qa` (small WSS, IO-bound)

Single-turn retrieval-augmented QA. Pure throughput test for control-plane components.

### 7.3 `code_agent` (large WSS, iterative)

Multi-turn coding tasks with tool use (read file, edit, test). Synthetic generator stressing memory/context handling. For *real* coding tasks with a deterministic oracle, use the `terminal_bench` workload (§7.8) instead — `code_agent` is the parameterizable synthetic complement, not a substitute.

### 7.4 `customer_support` (small WSS, high concurrency)

Intent classification + tool dispatch + reply. Models a real support agent. Stresses gateway routing under load.

### 7.5 `research` (large WSS, deep planning)

Search + read + synthesize multi-document research. Stresses long-context handling and citation tracking.

### 7.6 `mixed_realistic` (production blend)

70% RAG QA + 15% customer support + 10% code agent + 5% research. Closest to production load.

### 7.7 Trace replay

Replay recorded traces from production (Langfuse, OTEL exports, Bailian-style). Like AIPerf's `--trace-replay` mode.

### 7.8 `terminal_bench` (real tasks, deterministic oracle)

A first-class `WorkloadPlugin` wrapping Terminal-Bench: ~100 hand-crafted, human-verified terminal tasks (compile, debug, train, configure) each in a pinned Docker environment with a deterministic test-script oracle. Paired with a deterministic `QualityJudgePlugin` (the TB test scripts), this is the strongest available code-agent workload and it *solves* the quality-judging problem for that slice — the oracle is ground truth, not an LLM judge.

Integration topology is "Option A": AgentSysPerf drives the SUT while Terminal-Bench supplies tasks + oracle. This preserves per-component attribution but requires re-implementing the TB agent loop against AgentSysPerf's plugin contracts — a real, budgeted cost. Pin the dataset version (e.g. `terminal-bench-core` 0.1.1) or cross-run numbers are not comparable.

**Future work — benchmark adapters.** SWE-bench (same static + Dockerized + test-oracle mold, clean fit) and τ-bench / τ²-bench (interactive, needs a simulated-counterparty component and a state-diff judge; `pass^k` must stay distinct from `runs_per_arm`) are candidates for the same pattern. The durable abstraction is a `BenchmarkAdapter` = `WorkloadPlugin` + `QualityJudgePlugin` (+ optional simulated-counterparty), with `terminal_bench` as the reference template. Test-oracle benchmarks are the priority tier (they strengthen quality grounding); interactive and LLM-judged tiers come later. This is **scoped as future work, not yet built.**

---

## 8. WSS Routing as a Measured Hypothesis (Not a Built-In Assumption)

This section exists to protect the benchmark's credibility.

### The epistemic problem

Working-set-size (WSS) routing — classify a step by its data footprint, send it to the pool whose cache tier fits — is an attractive idea with a clean story. It is also, at this point, **a hypothesis**. The entire body of WSS work (node-local scheduling, semantic routing, split tool routing) rests on one unproven claim:

> *Routing by working-set-size improves cache locality, which improves task latency and cost, by more than the classification costs.*

If AgentSysPerf **assumed** this — baked WSS in as the default router, special-cased it in the harness, or compared real runs against a pre-baked WSS latency model — the benchmark would be confirming its own premise. A reviewer would catch the circularity immediately, and rightly discount every result.

### The resolution: routing strategy is a swept variable

WSS is not a feature of AgentSysPerf. It is **one registered `RoutingStrategyPlugin` among eleven**, with no privileged path through the harness. The benchmark's job is to find out whether it wins.

A routing strategy is a policy mapping `(request features, system state) → pool`. Strategies differ by *which inputs they consume* and *how*:

| Strategy | Routes on | Content-aware | Load-aware | Precedent |
|---|---|---|---|---|
| `single_pool` | nothing | ✗ | ✗ | null baseline |
| `round_robin` | rotation | ✗ | ✗ | default LB |
| `random` | RNG | ✗ | ✗ | trivial baseline |
| `least_loaded` | in-flight count | ✗ | ✓ | nginx least_conn |
| `power_of_two` | 2-sample load | ✗ | ✓ | near-optimal LB |
| `kv_affinity` | prefix-hash → warm cache | ✓ | partial | **vLLM production stack, SGLang** |
| `model_based` | model needed | ✓ | ✗ | multi-model gateways |
| `slo_aware` | deadline / SLA tier | ✓ | optional | premium-tier routing |
| `wss` | payload size → cache tier | ✓ | ✗ alone | **the hypothesis under test** |
| `semantic_slm` | learned from prompt | ✓ | optional | learned routers |
| `hybrid` | composition | ✓ | ✓ | **what production routers actually do** |

The controlled experiment lives in `routing_strategy_study.yaml`: workload, hardware, framework, backends, and classifier model are held constant; **only the routing strategy changes**. Any latency or cost difference is therefore attributable to the policy alone. WSS must beat `kv_affinity` and `hybrid` — the strong production baselines — *after its classification cost is charged* — to justify itself. The study is designed so it can return "WSS does not win," and that is an acceptable, publishable outcome.

### Three integrity safeguards built into the design

1. **Decision cost is charged.** Content-aware strategies (`wss`, `semantic_slm`, `kv_affinity`, `model_based`, `slo_aware`) must report non-zero `decision_cost_ms`, folded into `routing_decision_ms`. `round_robin` spends nothing deciding; WSS must not be flattered by hiding the price of classification.

2. **Missing signals abort, they do not degrade.** If a strategy's `required_signals` aren't available from the SUT, AgentSysPerf aborts the run with a clear error rather than silently running that strategy as random — which would otherwise hand WSS an unearned win.

3. **Ground-truth WSS labels remain a heuristic, openly.** `routing_accuracy` is scored against the `expected_wss` labels in the workload, which are byte-size heuristics, not validated optima. A high score means "agrees with the heuristic," not "provably correct." The honest caveat in §13 stays — softened, not deleted.

### Configurability

Everything above is config, not code change. `routing_strategy` is selectable per run; the WSS rule table is `configs/wss_rules.yaml` (auditable, not hard-coded); the study sweeps the full strategy set declaratively. Adding a twelfth strategy is a plugin registration, not a harness edit.

---

## 9. Optimization Profiles — Measured, Not Assumed (and the Strawman-Baseline Trap)

"Base vs Xeon-optimized" gets the same treatment as WSS routing: it is a **swept variable with engagement verification**, not a feature toggle. This section exists because base-vs-optimized is the single most cherry-picked number in CPU vendor benchmarking, and the framework has to structurally resist that.

### The contract

`OptimizationProfilePlugin` (contract 9) owns a declarative `OptimizationProfile` and **projects** it into the inference-backend and scheduler plugins via `project()`. The profile is decoupled from any specific backend/scheduler — it emits a `ProfileProjection` (backend config + scheduler config + the telemetry counters that must be checked) and the harness applies it. This is the projection layer that keeps "what optimization" separate from "which backend."

### The eight optimization axes

Each axis is independently togglable so a gain can be **attributed**, not hidden in one opaque "optimized" bundle:

| Layer | Axis | base | Xeon-optimized | Engagement counter |
|---|---|---|---|---|
| Backend | ISA | AVX-512 | AMX-TDPBSSD | `amx_active_cycle_ratio` |
| Backend | math library | reference | oneDNN / oneMKL | `onednn_kernel_dispatch_ratio` |
| Backend | runtime | vanilla | IPEX / OpenVINO | `openvino_infer_request_ratio` |
| Backend | quantization | FP32 | INT8/VNNI · INT4/AMX | (ISA counter + quality guard) |
| Scheduler | NUMA | unpinned | NUMA-local | `numa_remote_access_ratio` (max) |
| Scheduler | memory pages | 4K | 2M/1G hugepages | `hugepage_fault_ratio` |
| Scheduler | core isolation | shared | isolcpus+affinity | scheduling-jitter |
| Accelerator | offload | none | QAT / DSA / IAA | `accel_queue_depth_*` |

Reference profiles ship as `base`, `amx_only`, `amx_onednn`, `amx_numa_hugepages`, `openvino_xeon`, `full_xeon` (`optimization_profiles.py`). The controlled experiment is `base_vs_xeon_study.yaml`.

### The two integrity rules (this is the point of the section)

1. **The baseline is not a strawman.** `base` is a stock `pip install` reality — AVX-512, no oneDNN, FP32, unpinned — *not* an artificially crippled single-thread FP32 reference. `is_baseline=True` only for that arm, and a `baseline_purity_check` **invalidates the entire study** if the `base` arm shows any AMX activity (a secretly-optimized base would deflate the baseline and inflate every speedup). Every arm's full axis settings are published with results.

2. **A claimed optimization that didn't engage invalidates the arm.** "Built with AMX" is not evidence AMX ran — kernels silently fall back to AVX-512 on dtype/shape mismatch. After warmup, `verify_engaged()` reads real hardware counters via `HardwareTelemetryPlugin.read_engagement_counters()` and the arm **aborts** if a claimed optimization shows ~zero activity. A non-engaged arm is never reported as a valid "Xeon-optimized" data point. This is the same "verify, don't trust the label" discipline used for routing `required_signals`.

### The quality guard

`int8`/`int4` arms change numerics. The study reports `tokens_out_per_task` and `task_success_rate` *alongside* latency, specifically so a latency "win" that is actually a quality regression is visible. A latency gain with a success-rate drop is flagged, not celebrated.

### Configurability

Profile selection is per-run config. Adding a new optimization arm (e.g. a future ISA) is a plugin registration plus a study-YAML line — no harness edit. WSS routing and optimization profile are independent swept axes; a full factorial (`routing_strategy × optimization_profile`) is expressible but expensive, so the reference studies vary one axis at a time and note the interaction as future work.

---

## 10. Reference Plugins

Phase 1 ships with reference adapters for the most common stacks:

| Entity | Reference plugins |
|--------|-------------------|
| Agent framework | LangGraph, AutoGen, CrewAI, LlamaIndex agents, raw OpenAI-style chat loop |
| Gateway | Envoy (the config from earlier work), LiteLLM proxy, Kong AI Gateway, "no gateway" passthrough |
| Routing strategy | single_pool, round_robin, random, least_loaded, power_of_two, kv_affinity, model_based, slo_aware, wss, semantic_slm, hybrid (`routing_strategies.py`) |
| Optimization profile | base, amx_only, amx_onednn, amx_numa_hugepages, openvino_xeon, full_xeon (`optimization_profiles.py`) |
| Inference backend | vLLM, TGI, Triton, Ollama, **AIPerf-compatible** mock |
| Tools / RAG | Qdrant, Weaviate, Chroma, in-memory mock |
| Classifier | Rule-based (regex + size), 1B SLM (Granite-Micro), Llama-Guard for safety |
| Memory store | Redis, in-memory dict, SQLite |
| Hardware telemetry | Prometheus scraper, `dcgmi` for GPUs, `pcm` for Intel CPUs, `perf` (also the optimization-engagement verifier) |
| Quality judge | LLM-as-judge (GPT-4o, Claude), regex matcher, deterministic check, human queue |
| Workload | terminal_bench, legal_review, rag_qa, code_agent, customer_support, research, mixed_realistic, trace-replay |

---

## 11. CLI & Usage

Mirror AIPerf's CLI style for familiarity.

### Basic invocation

```bash
agentsysperf profile \
  --workload legal_review \
  --task-count 100 \
  --concurrency 10 \
  --agent-framework langgraph \
  --gateway envoy:http://gateway:8080 \
  --inference-backend vllm:http://vllm:8000 \
  --tools-rag qdrant:http://qdrant:6333 \
  --quality-judge llm:claude-sonnet \
  --output-format dashboard
```

### Comparative run (the killer feature)

```bash
agentsysperf compare \
  --workload legal_review \
  --task-count 200 \
  --variants variants.yaml \
  --metrics task_latency_e2e_ms,task_success_rate,cost_per_task_usd
```

Where `variants.yaml`:

```yaml
variants:
  - name: baseline_centralized
    gateway: envoy:8080
    scheduler: k8s_default
    classifier: rule_based

  - name: node_local_design
    gateway: envoy:8080
    scheduler: node_local
    classifier: slm_1b

  - name: ap_only_no_routing
    gateway: passthrough
    scheduler: k8s_default
    classifier: none
    hardware_label: xeon_ap
```

Output: side-by-side comparison table, statistical significance per metric.

### Trace replay

```bash
agentsysperf replay \
  --trace-file production_traces.jsonl \
  --target-stack my_stack.yaml
```

---

## 12. Comparison: Endpoint vs Stack Benchmarking

| Aspect | AIPerf | AgentSysPerf |
|--------|--------|-----------|
| **Unit of work** | One inference request | One agentic task (multi-step) |
| **Primary metrics** | TTFT, ITL, throughput | Task latency, success rate, cost/task |
| **Pluggable entities** | Endpoint type, dataset, transport, metric | All of AIPerf's + framework, gateway, scheduler, classifier, tools, hardware |
| **Workloads** | ShareGPT, SPEED-Bench, custom prompts | Task graphs, multi-step plans, multi-tool flows |
| **Quality measurement** | Output token count, response shape | Task completion, intent accuracy, routing correctness |
| **What it answers** | "How fast is my inference server?" | "How fast is my agent stack end-to-end? Where's the bottleneck?" |
| **Relationship** | AgentSysPerf depends on AIPerf-style backends being benchmarkable | AgentSysPerf can use AIPerf's measurements as the inference layer |

**The two are complementary, not competing.** AIPerf measures the inference component; AgentSysPerf measures everything around it.

---

## 13. Phased Roadmap

### Phase 1 — Core Framework (8-12 weeks)
- Three-plane architecture, ZMQ message bus (steal from AIPerf where possible)
- Plugin loader + ABC definitions for 8 entity types
- Two reference workloads: `rag_qa`, `legal_review`
- Reference plugins: LangGraph, Envoy, vLLM, Qdrant, rule-based classifier, LLM judge
- CSV/JSON/dashboard output
- **Deliverable:** Run a benchmark end-to-end against the worked example from our earlier work

### Phase 2 — Workload Library (4-6 weeks)
- 6 reference workloads (all of section 7)
- Trace replay format (canonical schema)
- Synthetic task generator with parameterized complexity
- Public dataset of ~10K reference tasks (open-source contribution)

### Phase 3 — Hardware-Aware (6-8 weeks)
- Additional hardware telemetry plugins (Intel `pcm` is the current path)
- Per-component CPU/GPU time attribution
- WSS classification accuracy ground-truth (manually labeled subset)
- CPU-vs-GPU comparison runs

### Phase 4 — Distributed (6-8 weeks)
- Multi-node benchmark coordination
- Federated scheduler test (the node-local design)
- Cross-node call attribution
- Gossip-protocol stress test

### Phase 5 — Community (ongoing)
- Open governance (a foundation or a vendor-neutral working group)
- Per-vendor reference results
- Quarterly published leaderboards

---

## 14. Honest Caveats

This is a **proposal**, not implemented code. Open questions:

⚠️ **Reproducibility in agentic systems is hard.** LLMs aren't deterministic even at temperature=0. Tool calls have side effects. Cache state varies across runs. We can seed RNGs and pin model versions, but **expect run-to-run variance of 10-20%** on quality metrics. The `multi-run-confidence` pattern from AIPerf is essential here.

⚠️ **Quality measurement is the hardest part.** LLM-as-judge has known issues (positional bias, sycophancy). Regex-based judges are too rigid. Human eval is expensive. We propose multiple judges + agreement scores, but **don't pretend this is solved**.

⚠️ **Ground-truth WSS labels are a heuristic, not validated optima.** The `expected_wss` labels in the reference workloads are assigned using the byte-size rubric in `configs/wss_rules.yaml`. This gives a principled, reproducible basis for `routing_accuracy` — but a step can be "medium" by payload size yet cheap if its KV prefix is cached. So `routing_accuracy` measures *agreement with the rubric*, not *provable optimality*. Establishing truly validated per-step optimal routing remains a research problem; the rubric makes the metric usable today without pretending it is solved.

⚠️ **The plugin abstractions assume cooperative SUTs.** If your gateway doesn't emit OTEL spans, AgentSysPerf can only measure black-box latency. Best results come from instrumented stacks.

⚠️ **No published cost numbers without your pricing.** Cost-per-task requires user-supplied pricing (per-token, per-GPU-hour, per-tool-call). We provide the schema; you provide the rates.

⚠️ **Comparison runs have a confound problem.** When swapping framework A for framework B, you also change versions, configs, defaults, network paths. Statistical significance helps but doesn't eliminate confounds. **Always publish full variant configs alongside results.**

⚠️ **Some metrics will be wrong at first.** Token economics for agentic flows is genuinely hard (continuation tokens, system prompt overhead, hidden retries). We'll iterate.

---

## 15. Why This Matters

**AIPerf made it possible to compare inference servers fairly.** Before AIPerf, every vendor cherry-picked their favorite metric, batch size, and sequence length. Standardization unlocked the market.

**Agentic stacks need the same treatment.** Today every framework's benchmarks are non-comparable: LangGraph publishes one number, CrewAI another, AutoGen a third. Cost claims are unverifiable. Scheduler trade-offs are undocumented. Routing strategy comparisons are vibes-based.

AgentSysPerf is the proposal to fix that.

---

## 16. Next Steps

1. **Validate motivation** with the user / community
2. **Build Phase 1 MVP** (4-person team, 8-12 weeks)
3. **Run on the worked example** from our earlier work as the first benchmark
4. **Publish reference results** for the Xeon-AP+SP configurations
5. **Open governance discussion** — should this live under a foundation or stay standalone?

---

## Appendix A — Relationship to Existing Tools

| Tool | What it does | Gap AgentSysPerf fills |
|------|--------------|---------------------|
| AIPerf | Single-endpoint LLM benchmark | Doesn't see multi-step agent workflows |
| MLPerf Inference | Vendor inference benchmarks | Same as AIPerf, plus rigid scenarios |
| HELM, lm-eval-harness | Model capability eval | Doesn't measure system performance |
| LangSmith / Langfuse | Production observability | Doesn't drive synthetic load |
| Promptfoo | Prompt regression testing | No load testing or stack-level measurement |
| AgentBench | Agent capability benchmarks | Quality-focused, not perf-focused |
| τ-bench, WebArena | Agent-in-environment benchmarks | Quality + completion, no perf attribution |

AgentSysPerf sits at the intersection: **stack-level performance + agent-aware metrics**.

## Appendix B — Glossary

- **SUT**: System Under Test — the user's agent stack being benchmarked
- **WSS**: Working Set Size — the data footprint of a single step (drives routing)
- **Task graph**: A DAG of agent steps with data dependencies
- **Pool**: A logical group of inference workers with similar capabilities (e.g., "xeon-ap-inference")
- **Cold route**: Routing decision that hits a backend with no warm cache for this prefix
- **Pluggable entity**: Any component the user can swap out (framework, gateway, scheduler, etc.)
