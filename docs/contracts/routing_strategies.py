#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""
=============================================================================
AgentSysPerf — Reference Routing Strategies
=============================================================================

Eleven reference implementations of RoutingStrategyPlugin. WSS is ONE of
them, deliberately placed in a field of credible alternatives so the
benchmark can MEASURE whether WSS routing actually wins — rather than
assuming it.

Each strategy declares `required_signals`. AgentSysPerf validates these against
what the SUT can actually supply BEFORE a run, and aborts with a clear error
if a strategy's inputs are unavailable — instead of silently degrading it to
random (which would corrupt the comparison).

Content-aware strategies (wss, kv_affinity, model_based, slo_aware,
semantic_slm) MUST report a non-zero decision_cost_ms. That cost is charged
back into the step's routing_decision_ms. Otherwise WSS / semantic routing
would look artificially cheaper than round_robin, which spends no compute
deciding.

CONFIDENCE NOTE:
- These are reference implementations for the benchmark, not production code.
- decision_cost_ms values for content-aware strategies are ESTIMATES and
  should be replaced with measured values on the target hardware before any
  published comparison. They are flagged inline.
- semantic_slm here is a stub: a real deployment runs a 1B classifier; the
  cost model below is illustrative only.
=============================================================================
"""

from __future__ import annotations
import itertools
import random
import time
from typing import Optional

from plugin_contracts import (
    RoutingStrategyPlugin,
    RoutingRequestFeatures,
    RoutingSystemState,
    RoutingDecision,
)


# =============================================================================
# Helper: pick first healthy pool as a safe default
# =============================================================================
def _healthy(available_pools: list[str], state: RoutingSystemState) -> list[str]:
    if state.healthy_pools:
        hp = [p for p in available_pools if p in state.healthy_pools]
        return hp or available_pools
    return available_pools


# =============================================================================
# 1. single_pool — null baseline
# =============================================================================
class SinglePoolStrategy(RoutingStrategyPlugin):
    """Everything goes to one pool. The control condition."""

    def __init__(self, pool: str = None):
        self._pool = pool

    @property
    def strategy_name(self) -> str:
        return "single_pool"

    @property
    def required_signals(self) -> set:
        return set()

    @property
    def is_content_aware(self) -> bool:
        return False

    async def decide(self, features, state, available_pools) -> RoutingDecision:
        pool = self._pool or _healthy(available_pools, state)[0]
        return RoutingDecision(
            target_pool=pool, decision_cost_ms=0.0,
            strategy_name=self.strategy_name, signals_used=[],
        )


# =============================================================================
# 2. round_robin — rotation, no awareness
# =============================================================================
class RoundRobinStrategy(RoutingStrategyPlugin):
    def __init__(self):
        self._counter = itertools.count()

    @property
    def strategy_name(self) -> str:
        return "round_robin"

    @property
    def required_signals(self) -> set:
        return set()

    @property
    def is_content_aware(self) -> bool:
        return False

    async def decide(self, features, state, available_pools) -> RoutingDecision:
        pools = _healthy(available_pools, state)
        idx = next(self._counter) % len(pools)
        return RoutingDecision(
            target_pool=pools[idx], decision_cost_ms=0.0,
            strategy_name=self.strategy_name, signals_used=[],
        )

    async def reset(self) -> None:
        self._counter = itertools.count()


# =============================================================================
# 3. random — RNG pick
# =============================================================================
class RandomStrategy(RoutingStrategyPlugin):
    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    @property
    def strategy_name(self) -> str:
        return "random"

    @property
    def required_signals(self) -> set:
        return set()

    @property
    def is_content_aware(self) -> bool:
        return False

    async def decide(self, features, state, available_pools) -> RoutingDecision:
        pools = _healthy(available_pools, state)
        return RoutingDecision(
            target_pool=self._rng.choice(pools), decision_cost_ms=0.0,
            strategy_name=self.strategy_name, signals_used=[],
        )


# =============================================================================
# 4. least_loaded — fewest in-flight (load-aware)
# =============================================================================
class LeastLoadedStrategy(RoutingStrategyPlugin):
    @property
    def strategy_name(self) -> str:
        return "least_loaded"

    @property
    def required_signals(self) -> set:
        return {"pool_loads"}

    @property
    def is_content_aware(self) -> bool:
        return False

    async def decide(self, features, state, available_pools) -> RoutingDecision:
        pools = _healthy(available_pools, state)
        if not state.pool_loads:
            # required signal missing → loud fallback flag, not silent random
            return RoutingDecision(
                target_pool=pools[0], decision_cost_ms=0.0,
                strategy_name=self.strategy_name, signals_used=[],
                fell_back=True,
            )
        chosen = min(pools, key=lambda p: state.pool_loads.get(p, 1.0))
        return RoutingDecision(
            target_pool=chosen, decision_cost_ms=0.0,
            strategy_name=self.strategy_name, signals_used=["pool_loads"],
        )


# =============================================================================
# 5. power_of_two — 2 random samples, pick lighter
# =============================================================================
class PowerOfTwoStrategy(RoutingStrategyPlugin):
    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    @property
    def strategy_name(self) -> str:
        return "power_of_two"

    @property
    def required_signals(self) -> set:
        return {"pool_loads"}

    @property
    def is_content_aware(self) -> bool:
        return False

    async def decide(self, features, state, available_pools) -> RoutingDecision:
        pools = _healthy(available_pools, state)
        if not state.pool_loads or len(pools) < 2:
            return RoutingDecision(
                target_pool=pools[0], decision_cost_ms=0.0,
                strategy_name=self.strategy_name, signals_used=[],
                fell_back=not state.pool_loads,
            )
        a, b = self._rng.sample(pools, 2)
        chosen = a if state.pool_loads.get(a, 1.0) <= state.pool_loads.get(b, 1.0) else b
        return RoutingDecision(
            target_pool=chosen, decision_cost_ms=0.0,
            strategy_name=self.strategy_name, signals_used=["pool_loads"],
        )


# =============================================================================
# 6. kv_affinity — prefix-hash → warm-cache pool (strong production baseline)
# =============================================================================
class KVAffinityStrategy(RoutingStrategyPlugin):
    """The baseline WSS must beat to justify itself. vLLM production stack /
    SGLang do exactly this. Load-aware tiebreak when no cache hit."""

    @property
    def strategy_name(self) -> str:
        return "kv_affinity"

    @property
    def required_signals(self) -> set:
        return {"prompt_prefix_hash", "pool_kv_cache_keys"}

    @property
    def is_content_aware(self) -> bool:
        return True

    async def decide(self, features, state, available_pools) -> RoutingDecision:
        pools = _healthy(available_pools, state)
        t0 = time.perf_counter()

        if features.prompt_prefix_hash is None or state.pool_kv_cache_keys is None:
            return RoutingDecision(
                target_pool=pools[0],
                decision_cost_ms=(time.perf_counter() - t0) * 1000,
                strategy_name=self.strategy_name, signals_used=[],
                fell_back=True,
            )

        # warm-cache pool wins
        for p in pools:
            if features.prompt_prefix_hash in state.pool_kv_cache_keys.get(p, set()):
                return RoutingDecision(
                    target_pool=p,
                    decision_cost_ms=(time.perf_counter() - t0) * 1000,
                    strategy_name=self.strategy_name,
                    signals_used=["prompt_prefix_hash", "pool_kv_cache_keys"],
                )
        # no hit → least-loaded tiebreak if available, else first
        if state.pool_loads:
            chosen = min(pools, key=lambda p: state.pool_loads.get(p, 1.0))
        else:
            chosen = pools[0]
        return RoutingDecision(
            target_pool=chosen,
            decision_cost_ms=(time.perf_counter() - t0) * 1000,
            strategy_name=self.strategy_name,
            signals_used=["prompt_prefix_hash", "pool_kv_cache_keys"],
        )


# =============================================================================
# 7. model_based — route by which model the step needs
# =============================================================================
class ModelBasedStrategy(RoutingStrategyPlugin):
    @property
    def strategy_name(self) -> str:
        return "model_based"

    @property
    def required_signals(self) -> set:
        return {"model_required", "pool_models"}

    @property
    def is_content_aware(self) -> bool:
        return True

    async def decide(self, features, state, available_pools) -> RoutingDecision:
        pools = _healthy(available_pools, state)
        t0 = time.perf_counter()
        if not features.model_required or not state.pool_models:
            return RoutingDecision(
                target_pool=pools[0],
                decision_cost_ms=(time.perf_counter() - t0) * 1000,
                strategy_name=self.strategy_name, signals_used=[],
                fell_back=True,
            )
        for p in pools:
            if features.model_required in state.pool_models.get(p, set()):
                return RoutingDecision(
                    target_pool=p,
                    decision_cost_ms=(time.perf_counter() - t0) * 1000,
                    strategy_name=self.strategy_name,
                    signals_used=["model_required", "pool_models"],
                )
        return RoutingDecision(
            target_pool=pools[0],
            decision_cost_ms=(time.perf_counter() - t0) * 1000,
            strategy_name=self.strategy_name,
            signals_used=["model_required", "pool_models"], fell_back=True,
        )


# =============================================================================
# 8. slo_aware — route by remaining deadline / SLA tier
# =============================================================================
class SLOAwareStrategy(RoutingStrategyPlugin):
    """Tight deadline / premium tier → fast pool; relaxed → cheap pool.
    Pool speed ranking supplied via config (fast_pools ordered fastest-first)."""

    def __init__(self, fast_pools: list[str] = None, tight_deadline_s: float = 2.0):
        self._fast_pools = fast_pools or []
        self._tight = tight_deadline_s

    @property
    def strategy_name(self) -> str:
        return "slo_aware"

    @property
    def required_signals(self) -> set:
        return {"deadline_s"}  # sla_tier optional

    @property
    def is_content_aware(self) -> bool:
        return True

    async def decide(self, features, state, available_pools) -> RoutingDecision:
        pools = _healthy(available_pools, state)
        t0 = time.perf_counter()
        ranked = [p for p in self._fast_pools if p in pools] or pools
        deadline = features.deadline_s
        premium = (features.sla_tier == "premium")
        if deadline is None and not premium:
            return RoutingDecision(
                target_pool=pools[0],
                decision_cost_ms=(time.perf_counter() - t0) * 1000,
                strategy_name=self.strategy_name, signals_used=[],
                fell_back=True,
            )
        if premium or (deadline is not None and deadline <= self._tight):
            chosen = ranked[0]                    # fastest
        else:
            chosen = ranked[-1]                   # cheapest/slowest acceptable
        return RoutingDecision(
            target_pool=chosen,
            decision_cost_ms=(time.perf_counter() - t0) * 1000,
            strategy_name=self.strategy_name,
            signals_used=["deadline_s", "sla_tier"],
        )


# =============================================================================
# 9. wss — THE HYPOTHESIS UNDER TEST
# =============================================================================
class WSSStrategy(RoutingStrategyPlugin):
    """Working-set-size routing. Payload bytes → cache tier → pool.

    Rule table is loaded from configs/wss_rules.yaml so it is auditable and
    not hard-coded as a privileged default. This is the strategy the whole
    study exists to evaluate — it gets NO special treatment in the harness.

    NOTE: decision_cost_ms below is the cost of the rule lookup only. If a
    deployment uses an SLM to estimate WSS instead of byte size, use
    semantic_slm (strategy 10) or hybrid (11) — do not pretend SLM cost is
    zero here.
    """

    def __init__(self, rules: list[dict] = None):
        # rules: ordered list of {max_bytes, wss_class, pool,
        #                          override_task_types?, override_pool?}
        self._rules = rules or self._default_rules()

    @staticmethod
    def _default_rules() -> list[dict]:
        # Mirrors configs/wss_rules.yaml. Kept in sync intentionally; the YAML
        # is the source of truth for runs, this is the test fallback.
        return [
            {"max_bytes": 1024,            "wss_class": "tiny",
             "pool": "control"},
            {"max_bytes": 32 * 1024,       "wss_class": "small",
             "pool": "rerank"},
            {"max_bytes": 32 * 1024 * 1024, "wss_class": "medium",
             "pool": "inference"},
            {"max_bytes": None,            "wss_class": "large",
             "pool": "gpu_coord",
             "override_task_types": ["api_call", "tool_dispatch"],
             "override_pool": "control"},
        ]

    @property
    def strategy_name(self) -> str:
        return "wss"

    @property
    def required_signals(self) -> set:
        return {"payload_bytes"}

    @property
    def is_content_aware(self) -> bool:
        return True

    async def decide(self, features, state, available_pools) -> RoutingDecision:
        pools = _healthy(available_pools, state)
        t0 = time.perf_counter()
        n = features.payload_bytes
        chosen_logical = None
        for rule in self._rules:
            mb = rule["max_bytes"]
            if mb is None or n < mb:
                chosen_logical = rule["pool"]
                if (rule.get("override_task_types")
                        and features.task_type in rule["override_task_types"]):
                    chosen_logical = rule["override_pool"]
                break
        # map logical pool name → an available concrete pool (substring match)
        target = next(
            (p for p in pools if chosen_logical and chosen_logical in p),
            pools[0],
        )
        # ESTIMATE: rule-table lookup ~ a few microseconds. Replace with a
        # measured value before any published comparison.
        cost_ms = (time.perf_counter() - t0) * 1000
        return RoutingDecision(
            target_pool=target, decision_cost_ms=cost_ms,
            strategy_name=self.strategy_name, signals_used=["payload_bytes"],
        )


# =============================================================================
# 10. semantic_slm — tiny model predicts route (STUB cost model)
# =============================================================================
class SemanticSLMStrategy(RoutingStrategyPlugin):
    """A 1B classifier predicts the route from prompt semantics. Here it is
    stubbed: it delegates to an injected `predict_fn` and adds a configurable
    `slm_cost_ms` representing the SLM forward pass. That cost is REAL and is
    charged — semantic routing is not free."""

    def __init__(self, predict_fn=None, slm_cost_ms: float = 8.0):
        # slm_cost_ms ESTIMATE: ~8 ms for a 1B INT4 forward on a few CPU cores.
        # Replace with measured value on target hardware.
        self._predict = predict_fn or (lambda f: None)
        self._slm_cost_ms = slm_cost_ms

    @property
    def strategy_name(self) -> str:
        return "semantic_slm"

    @property
    def required_signals(self) -> set:
        return set()  # operates on the prompt the agent already has

    @property
    def is_content_aware(self) -> bool:
        return True

    async def decide(self, features, state, available_pools) -> RoutingDecision:
        pools = _healthy(available_pools, state)
        pred = self._predict(features)
        target = next((p for p in pools if pred and pred in p), pools[0])
        return RoutingDecision(
            target_pool=target,
            decision_cost_ms=self._slm_cost_ms,   # charged, not free
            strategy_name=self.strategy_name,
            signals_used=["prompt_semantics"],
            fell_back=(pred is None),
        )


# =============================================================================
# 11. hybrid — composition (what production routers actually do)
# =============================================================================
class HybridStrategy(RoutingStrategyPlugin):
    """Ordered fallthrough: model capability filter → kv_affinity →
    least_loaded tiebreak. Configurable stage list. This is the realistic
    production baseline; if `hybrid` beats `wss`, that is a finding worth
    publishing, not hiding."""

    def __init__(self, stages: list[str] = None):
        self._stages = stages or ["model_based", "kv_affinity", "least_loaded"]
        self._impls = {
            "model_based": ModelBasedStrategy(),
            "kv_affinity": KVAffinityStrategy(),
            "least_loaded": LeastLoadedStrategy(),
        }

    @property
    def strategy_name(self) -> str:
        return "hybrid"

    @property
    def required_signals(self) -> set:
        s: set = set()
        for st in self._stages:
            s |= self._impls[st].required_signals
        return s

    @property
    def is_content_aware(self) -> bool:
        return True

    async def decide(self, features, state, available_pools) -> RoutingDecision:
        t0 = time.perf_counter()
        used: list[str] = []
        for st in self._stages:
            d = await self._impls[st].decide(features, state, available_pools)
            used += d.signals_used
            if not d.fell_back:
                return RoutingDecision(
                    target_pool=d.target_pool,
                    decision_cost_ms=(time.perf_counter() - t0) * 1000,
                    strategy_name=self.strategy_name,
                    signals_used=used,
                )
        # all stages fell back
        pools = _healthy(available_pools, state)
        return RoutingDecision(
            target_pool=pools[0],
            decision_cost_ms=(time.perf_counter() - t0) * 1000,
            strategy_name=self.strategy_name, signals_used=used,
            fell_back=True,
        )


# =============================================================================
# REGISTRY — what entry_points would expose under agentsysperf.routing_strategy
# =============================================================================
REFERENCE_STRATEGIES = {
    "single_pool":  SinglePoolStrategy,
    "round_robin":  RoundRobinStrategy,
    "random":       RandomStrategy,
    "least_loaded": LeastLoadedStrategy,
    "power_of_two": PowerOfTwoStrategy,
    "kv_affinity":  KVAffinityStrategy,
    "model_based":  ModelBasedStrategy,
    "slo_aware":    SLOAwareStrategy,
    "wss":          WSSStrategy,
    "semantic_slm": SemanticSLMStrategy,
    "hybrid":       HybridStrategy,
}
