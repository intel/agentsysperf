#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Run driver (P11) — the shared benchmark-run loop behind `agentsysperf run`.

Folds the canonical runner (scripts/run_tb2_dashboard_demo.py) into a reusable
function built on the unified storage path: RunContext(result_store=open()) +
persist_run, one run_id across store/Prometheus/Langfuse, per-task timeout with
error isolation, dual-write (JSON + store).

Two benchmark families:
- synthetic_cpu: no LLM, no network, no key — uses a noop invoker (CI-testable).
- terminal-bench: loads tasks from Harbor, drives LiteLLM; optionally wrapped in
  a ReplayProxy (--record / --replay) so the costly/nondeterministic LLM calls
  are journalled once and served free thereafter, while hardware is still
  measured on the current silicon.
"""
from __future__ import annotations

import concurrent.futures as cf
import logging
import os
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

from src.protocols import discover_analyzers, discover_measurements
from src.runner import RunContext, track_span
import tempfile

# Scratch root for run artifacts. gettempdir() honours TMPDIR and falls
# back to "/tmp", so this resolves to the same paths as the hardcoded
# /tmp literals it replaced while no longer pinning output to a
# world-writable directory on systems that configure TMPDIR elsewhere.
_TMP = tempfile.gettempdir()

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """Transport configuration for one benchmark run.

    ``dataset_name``, ``dataset_ref``, and ``preresolved_tasks`` are parent
    to worker transport fields used by ``run-streams`` so Harbor is resolved
    once before workers start. ``adapter_kwargs`` carries internal adapter
    state such as the task-sized resource budget. ``proxy_port`` selects the
    per-worker replay proxy port. ``stream_metadata`` records the admission
    decision and scheduling mode in run provenance.
    """

    benchmark: str = "terminal-bench"
    model: str = "gpt-4o-mini"
    num_tasks: Optional[int] = None          # None + full=False -> a small default
    full: bool = False
    tasks: Optional[Sequence[str]] = None     # explicit task names (include=)
    # 30 turns truncated real work: a 10-task Terminal-Bench run had 7 of 8
    # failures stop at exactly the cap, mid-task, and scored 2/10. Harbor's own
    # reference agent (terminus_2) defaults to 1_000_000 turns and warns that a
    # cap "artificially limited" the run. 200 is a compromise: it clears the
    # measured need (passes used 18-50 turns) without letting a thrashing agent
    # run unbounded. Raise it rather than accept a truncated score.
    max_turns: int = 200
    # Sized from the same run: 7.1s/turn median, so 200 turns ~ 24 min typical
    # and ~95 min at the slowest observed rate. 420s could not even fit 60
    # median turns.
    timeout_s: int = 3600                      # per-task wall-clock cap
    run_id: Optional[str] = None
    output_dir: Optional[Path] = None
    replay_mode: str = "off"                  # off | record | replay
    fixture: Optional[Path] = None
    upstream: Optional[str] = None            # record: real LLM base URL to forward to
                                              # (defaults to $OPENAI_BASE_URL)
    dataset_name: str = "terminal-bench/terminal-bench-2"
    dataset_ref: str = "latest"
    adapter_kwargs: Optional[Mapping[str, Any]] = None
    preresolved_tasks: Optional[Sequence[Any]] = None
    proxy_port: Optional[int] = None
    stream_metadata: Optional[Mapping[str, Any]] = None


@dataclass
class RunSummary:
    run_id: str
    total: int
    passed: int
    records: int
    db_path: str
    coverage: Optional["RunCoverage"] = None
    execution_failures: int = 0
    oracle_failures: int = 0


@dataclass
class RunCoverage:
    """Which measurements produced data and which analyzers spoke, and why not.

    A silently-skipped probe is indistinguishable from one that ran unless the
    run says so: an `emon` with no SEP driver and an `emon` that collected 629
    metrics both end in "9 records". This is computed from what actually
    reached ``ctx.records`` — not from what ``discover_measurements()``
    returned — so it cannot claim a layer that produced nothing.
    """

    measurements_active: List[str]
    measurements_silent: List[tuple]     # (name, reason)
    analyzers_emitted: List[str]
    analyzers_silent: List[tuple]        # (name, reason)

    @property
    def measurements_total(self) -> int:
        return len(self.measurements_active) + len(self.measurements_silent)

    @property
    def analyzers_total(self) -> int:
        return len(self.analyzers_emitted) + len(self.analyzers_silent)

    def as_lines(self) -> List[str]:
        """Two human-readable lines: one per plugin family."""
        def _fmt(silent: List[tuple]) -> str:
            return "; ".join(f"{n}: {r}" for n, r in silent)

        out = [
            f"measurements {len(self.measurements_active)}/{self.measurements_total} active"
            + (f" ({_fmt(self.measurements_silent)})" if self.measurements_silent else ""),
            f"analyzers {len(self.analyzers_emitted)}/{self.analyzers_total} emitted"
            + (f" ({_fmt(self.analyzers_silent)})" if self.analyzers_silent else ""),
        ]
        return out


def _measurement_skip_reason(probe: Any) -> str:
    """Best available explanation for a probe that emitted no records.

    Probes signal unavailability inconsistently — ``_available=False`` (l3_perf,
    emon, perfspect), or no flag at all (the l1_* probes, which always work). We
    read the flag when present and otherwise report the honest thing: it was
    asked and produced nothing.
    """
    if getattr(probe, "_available", True) is False:
        return "unavailable on this host (tool/driver/permission)"
    return "no records for any span"


_SILENT_ANALYZER_HINTS = {
    "phase_profiler": "no phase-tagged spans (synthetic workloads have none; use terminal-bench or swe-bench)",
    "scaling": "single-run scope (scaling needs a multi-density sweep: agentsysperf sweep run)",
}


def _analyzer_skip_reason(
    analyzer: Any, layers_present: set, failed: Optional[str],
) -> str:
    """Why an analyzer emitted no verdict."""
    if failed is not None:
        return f"failed: {failed}"
    required = getattr(analyzer, "input_layers", None)
    if required:
        missing = sorted(set(required) - layers_present)
        if missing:
            return f"needs layer(s) {', '.join(missing)}"
    name = getattr(analyzer, "name", "")
    hint = _SILENT_ANALYZER_HINTS.get(name)
    if hint:
        return hint
    return "ran but emitted no verdict"


def _compute_coverage(
    *,
    measurements: dict,
    analyzers: dict,
    records: Sequence[Any],
    verdicts: Sequence[Any],
    analyzer_errors: dict,
) -> RunCoverage:
    """Reconcile discovered plugins against what they actually produced."""
    layers_present = {r.layer for r in records}

    active, silent = [], []
    for name, probe in sorted(measurements.items()):
        if getattr(probe, "layer", None) in layers_present:
            active.append(name)
        else:
            silent.append((name, _measurement_skip_reason(probe)))

    spoke = {getattr(v, "analyzer_name", None) for v in verdicts}
    emitted, quiet = [], []
    for name, an in sorted(analyzers.items()):
        if name in spoke or getattr(an, "name", None) in spoke:
            emitted.append(name)
        else:
            quiet.append((
                name,
                _analyzer_skip_reason(an, layers_present, analyzer_errors.get(name)),
            ))

    return RunCoverage(
        measurements_active=active, measurements_silent=silent,
        analyzers_emitted=emitted, analyzers_silent=quiet,
    )


def _run_task_params(adapter: Any) -> frozenset:
    """Parameter names accepted by adapter.run_task (cached per class)."""
    import inspect
    cache = _run_task_params.__dict__.setdefault("_cache", {})
    cls = type(adapter)
    if cls not in cache:
        cache[cls] = frozenset(inspect.signature(adapter.run_task).parameters)
    return cache[cls]


def _resolve_run_id(cfg: RunConfig, stamp: int) -> str:
    if cfg.run_id:
        return cfg.run_id
    return f"{cfg.benchmark.replace('/', '-')}_{stamp}"


def _uses_llm(cfg: RunConfig) -> bool:
    """Whether this run drives a real model.

    Single source of truth for both the invoker choice and the recorded
    metadata: ``RunConfig.model`` carries a default whether or not a model is
    ever called, so recording it unconditionally stamped every synthetic run
    with a model it never invoked — visible in `db ls` and every report.
    """
    return cfg.benchmark != "synthetic_cpu"


def _build_invoker(cfg: RunConfig):
    """Synthetic needs no LLM; terminal-bench drives LiteLLM."""
    if not _uses_llm(cfg):
        from src.testing.noop_invoker import NoOpAgentInvoker
        return NoOpAgentInvoker()
    from src.agent_loops.litellm_invoker import LiteLLMAgentInvoker
    return LiteLLMAgentInvoker(model=cfg.model, max_turns=cfg.max_turns, temperature=0.0)


_harbor_task_cache: dict[tuple[Any, ...], list[Any]] = {}


def _load_harbor_tasks_cached(
    *,
    dataset_name: str,
    ref: str,
    task_names: Optional[Sequence[str]],
    limit: Optional[int],
) -> list[Any]:
    """Resolve Harbor metadata once per process for identical task requests."""
    from src.benchmarks.terminal_bench.harbor_loader import (
        load_tasks_from_harbor_registry,
    )

    key = (dataset_name, ref, tuple(task_names) if task_names else None, limit)
    if key not in _harbor_task_cache:
        _harbor_task_cache[key] = list(
            load_tasks_from_harbor_registry(
                dataset_name=dataset_name,
                ref=ref,
                task_names=list(task_names) if task_names else None,
                limit=limit,
            )
        )
    return _harbor_task_cache[key]


def _build_adapter_and_tasks(cfg: RunConfig):
    """Return (adapter, task_specs). Selection: --tasks > --num-tasks > --full."""
    if cfg.benchmark == "synthetic_cpu":
        from src.benchmarks.synthetic_cpu import SyntheticCpuAdapter
        adapter = SyntheticCpuAdapter()
        limit = None if cfg.full else (cfg.num_tasks or None)
        specs = list(adapter.list_tasks(include=cfg.tasks, limit=limit))
        return adapter, specs

    if cfg.benchmark in ("terminal-bench", "terminal_bench"):
        from src.benchmarks.terminal_bench import TerminalBenchAdapter
        names = (
            [f"terminal-bench/{name.split('/')[-1]}" for name in cfg.tasks]
            if cfg.tasks
            else None
        )
        limit = None if cfg.full else (cfg.num_tasks or 5)
        harbor_tasks = list(cfg.preresolved_tasks) if cfg.preresolved_tasks is not None else (
            _load_harbor_tasks_cached(
                dataset_name=cfg.dataset_name,
                ref=cfg.dataset_ref,
                task_names=names,
                limit=None if names else limit,
            )
        )
        adapter = TerminalBenchAdapter(
            dataset_loader=lambda: harbor_tasks,
            **dict(cfg.adapter_kwargs or {}),
        )
        specs = list(adapter.list_tasks(limit=None if names else limit))
        if names:
            by_name = {spec.id.split("/")[-1]: spec for spec in specs}
            requested = [name.split("/")[-1] for name in names]
            missing = [name for name in requested if name not in by_name]
            if missing:
                raise ValueError(f"Terminal-Bench task(s) not found: {missing}")
            specs = [by_name[name] for name in requested]
        return adapter, specs

    raise ValueError(f"benchmark {cfg.benchmark!r} not supported by `agentsysperf run` yet")


def _read_first_line(path: str) -> Optional[str]:
    """Read a single sysfs/procfs value, or None if unreadable."""
    try:
        with open(path) as f:
            return f.read().strip()
    except (IOError, OSError):
        return None


def _env_fingerprint() -> dict:
    """Capture the tuning knobs that change measured numbers.

    Recorded so a run can be interpreted (and compared) later: a benchmark
    collected under governor=powersave is not comparable to one under
    performance, and perf_event_paranoid silently determines whether hardware
    counters were available at all. Warn-only — a misconfigured host still
    runs, but the run says so.
    """
    env: dict = {}

    governor = _read_first_line(
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
    )
    if governor:
        env["cpu_governor"] = governor
        if governor != "performance":
            logger.warning(
                "cpu_governor=%s (not 'performance') — frequency will vary "
                "during measurement. Run `agentsysperf preflight --fix` to set "
                "it. Recorded in run provenance.",
                governor,
            )

    paranoid = _read_first_line("/proc/sys/kernel/perf_event_paranoid")
    if paranoid is not None:
        env["perf_event_paranoid"] = paranoid
        try:
            if int(paranoid) > 1:
                logger.warning(
                    "perf_event_paranoid=%s (need <=1) — hardware counter "
                    "measurements (L3, PerfSpect) will collect nothing. "
                    "Recorded in run provenance.",
                    paranoid,
                )
        except ValueError:
            pass

    watchdog = _read_first_line("/proc/sys/kernel/nmi_watchdog")
    if watchdog is not None:
        env["nmi_watchdog"] = watchdog

    no_turbo = _read_first_line("/sys/devices/system/cpu/intel_pstate/no_turbo")
    if no_turbo is not None:
        env["turbo_enabled"] = no_turbo == "0"

    return env


def _run_metadata(cfg: RunConfig, stamp: int) -> dict:
    sku = None
    platform_meta: dict = {}
    try:
        from src.platform.detect import detect_platform
        p = detect_platform()
        sku = p.model_name
        platform_meta = {
            "microarchitecture": p.microarchitecture,
            "uarch_source": p.uarch_source,
            "cpu_family": p.cpu_family,
            "cpu_model": p.cpu_model,
            "cpu_stepping": p.cpu_stepping,
            "physical_cores": p.physical_cores,
            "logical_cpus": p.logical_cpus,
            "sockets": p.sockets,
            "numa_nodes": p.numa_nodes,
            "dram_bw_total_gbs": p.dram_bw_total_gbs or None,
            "dram_bw_source": p.dram_bw_source,
        }
    except Exception:
        logger.debug("platform detection failed for run metadata", exc_info=True)

    env = {}
    try:
        env = _env_fingerprint()
    except Exception:
        logger.debug("environment fingerprint failed", exc_info=True)

    stream_scheduling = dict(cfg.stream_metadata or {}) or None
    budget = (cfg.adapter_kwargs or {}).get("resource_budget")
    numa_policy = None
    if budget is not None and budget.cpuset:
        numa_policy = f"cpuset:{budget.cpuset_str()}"
    elif stream_scheduling and stream_scheduling.get("mode") == "unpinned":
        numa_policy = "unpinned"
    return {
        "start_time": stamp,
        "benchmark_id": cfg.benchmark.replace("_", "-"),
        "model": cfg.model if _uses_llm(cfg) else None,
        "hardware_sku": sku,
        "owner_id": os.environ.get("USER") or os.environ.get("USERNAME"),
        "owner_kind": "os_user",
        "host_id": socket.gethostname(),
        "numa_policy": numa_policy,
        "stream_scheduling": stream_scheduling,
        "platform": platform_meta or None,
        "environment": env or None,
    }


def run_benchmark(cfg: RunConfig, *, store: Any = None, stamp: Optional[int] = None) -> RunSummary:
    """Execute a benchmark run, persisting to the canonical store.

    ``store`` defaults to ResultStore.open(); ``stamp`` is an epoch-seconds
    timestamp (injectable for deterministic run_ids in tests).
    """
    stamp = stamp if stamp is not None else int(time.time())
    if store is None:
        from src.storage.sqlite_store import SQLiteResultStore
        store = SQLiteResultStore.open()

    run_id = _resolve_run_id(cfg, stamp)
    adapter, specs = _build_adapter_and_tasks(cfg)
    invoker = _build_invoker(cfg)
    measurements = discover_measurements()
    analyzers = discover_analyzers()

    out = cfg.output_dir or Path(f"{_TMP}/agentsysperf_scratch/{run_id}")
    ctx = RunContext(
        run_id=run_id,
        measurements=list(measurements.values()),
        output_dir=out,
        result_store=store,
        run_metadata=_run_metadata(cfg, stamp),
    )

    # Optional replay/record proxy: point litellm at it via env (litellm reads
    # OPENAI_API_BASE/OPENAI_BASE_URL from the environment — no invoker change).
    proxy_cm = _maybe_proxy(cfg)

    task_results: List = []
    passed = 0
    execution_failures = 0
    oracle_failures = 0
    coverage: Optional[RunCoverage] = None
    ex = cf.ThreadPoolExecutor(max_workers=1)
    _mode_label = {"record": "recording", "replay": "replaying", "off": "live"}.get(
        cfg.replay_mode, cfg.replay_mode)
    _no_llm_benchmarks = {"synthetic_cpu", "synthetic-cpu"}
    _uses_llm = cfg.benchmark not in _no_llm_benchmarks

    # One-shot PMU contention check (subprocess.run is ~5ms).
    try:
        _ps = subprocess.run(
            ["pgrep", "-a", "-f", "perf stat.*-a"],
            capture_output=True, text=True, timeout=2,
        )
        _other_perf = [
            ln for ln in _ps.stdout.splitlines()
            if str(os.getpid()) not in ln
        ]
        if _other_perf:
            print(
                f"  WARNING: {len(_other_perf)} other 'perf stat -a' process(es) "
                f"detected — L3 counters may be unreliable (PMU contention).",
                flush=True,
            )
            print(
                f"  Fix: stop the other exclusive-PMU collector(s). "
                f"PIDs: {', '.join(ln.split()[0] for ln in _other_perf[:3])}",
                flush=True,
            )
    except Exception:
        pass

    print(
        f"Running {cfg.benchmark} ({len(specs)} task{'s' if len(specs) != 1 else ''}, "
        f"mode={_mode_label})",
        flush=True,
    )
    if _uses_llm:
        _llm_target = (
            os.environ.get("OPENAI_API_BASE")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1 (default)"
        )
        if cfg.replay_mode == "replay":
            _llm_target = "local replay proxy (no LLM calls)"
        print(f"  LLM endpoint: {_llm_target}", flush=True)
        print(f"  Model: {cfg.model}", flush=True)
    try:
        with proxy_cm as proxy:
            if proxy is not None:
                os.environ.update(proxy.env)
                # Two hops with OPPOSITE proxy needs:
                #   1. litellm (here) -> local proxy (127.0.0.1): must BYPASS the
                #      corporate proxy.
                #   2. proxy subprocess -> upstream LLM (maybe external, e.g.
                #      Bedrock): must USE the corporate proxy (no direct egress).
                # So we KEEP HTTP(S)_PROXY (hop 2, in the subprocess) and only
                # ensure NO_PROXY lists the loopback host explicitly (hop 1).
                # The openai/httpx client honors an explicit host in NO_PROXY
                # even when it ignores the CIDR form (127.0.0.0/8).
                _hostparts = {"127.0.0.1", "localhost", proxy.host}
                _existing = os.environ.get("NO_PROXY", "")
                os.environ["NO_PROXY"] = ",".join(
                    [p for p in (_existing.split(",") if _existing else []) if p]
                    + [h for h in _hostparts if h not in _existing]
                )
                os.environ["no_proxy"] = os.environ["NO_PROXY"]
            _needs_docker = cfg.benchmark in ("terminal-bench", "terminal_bench")
            if _needs_docker and len(specs) > 0:
                print("  (each task: Docker build → agent turns → score; first task is slowest)",
                      flush=True)
            with ctx:
                for i, spec in enumerate(specs, 1):
                    short = spec.id.split("/")[-1]
                    print(f"  [{i}/{len(specs)}] {short} ...", end="", flush=True)
                    logger.info("▶ %s", short)
                    t0 = time.time()
                    result = None
                    with track_span(ctx, spec.id, kind=cfg.benchmark, node_id=short):
                        # run_context is TerminalBenchAdapter-specific (it threads
                        # StepTraces back); synthetic_cpu.run_task doesn't accept
                        # it. Pass only what the adapter's signature declares.
                        kw = {"agent_invoker": invoker}
                        if "run_context" in _run_task_params(adapter):
                            kw["run_context"] = ctx
                        fut = ex.submit(adapter.run_task, spec, **kw)
                        task_execution_failed = False
                        try:
                            result = fut.result(timeout=cfg.timeout_s)
                        except cf.TimeoutError as e:
                            # On Python 3.11+, cf.TimeoutError IS TimeoutError,
                            # so SDK-raised TimeoutError lands here too. Distinguish
                            # by checking if the future actually timed out.
                            if fut.done():
                                # Future completed but raised TimeoutError internally
                                # (e.g. sandbox setup timeout) — not our wall-clock cap.
                                logger.warning("  ERROR (%s) — span still captured", e)
                            else:
                                logger.warning("  TIMEOUT after %ds — span still captured", cfg.timeout_s)
                            task_execution_failed = True
                        except Exception as e:  # one bad task != dead run
                            logger.warning("  ERROR (%s) — span still captured", e)
                            task_execution_failed = True
                    dur = time.time() - t0
                    _status = "PASS" if (result and result.passed) else "FAIL"
                    print(f" {_status} ({dur:.1f}s)", flush=True)
                    ok = bool(result and result.passed)
                    passed += ok
                    if task_execution_failed or result is None or result.error:
                        execution_failures += 1
                    elif (result.extra or {}).get("oracle_run") and not result.passed:
                        oracle_failures += 1
                    measured = (result.extra or {}).get("measured_s") if result else None
                    elapsed = (result.extra or {}).get("elapsed_s") if result else None
                    task_results.append((
                        (result.task_id if result else spec.id),
                        {"passed": ok, "duration_s": measured if measured is not None else dur,
                         "elapsed_s": elapsed,
                         "reward": (result.reward if result else 0.0),
                         "num_turns": (result.extra or {}).get("num_turns") if result else None,
                         # The adapter reports this in extra; omitting it here left
                         # task_results.num_commands NULL for every run ever stored,
                         # which reads as "the agent ran no commands" rather than
                         # "nobody wrote the value down". The counts were only
                         # recoverable by counting */turn_*_cmd spans.
                         "num_commands": (result.extra or {}).get("num_commands") if result else None},
                    ))
            # ctx.stop() (on __exit__) already dual-wrote run+measurements+spans
            # via persist_run. Persist the task results + analyzer verdicts too
            # (the runner doesn't own those) in one transaction.
            analysis = []
            analyzer_errors: dict = {}
            for name, an in analyzers.items():
                try:
                    analysis.extend(an.analyze(ctx.records))
                except Exception as e:
                    logger.warning("analyzer %s failed: %s", name, e)
                    analyzer_errors[name] = str(e)
            coverage = _compute_coverage(
                measurements=measurements, analyzers=analyzers,
                records=ctx.records, verdicts=analysis,
                analyzer_errors=analyzer_errors,
            )
            store.persist_run(
                run_id=run_id,
                metadata={**_run_metadata(cfg, stamp),
                          "end_time": int(time.time()),
                          "total_tasks": len(specs), "passed_tasks": passed,
                          "execution_failures": execution_failures,
                          "oracle_failures": oracle_failures,
                          "status": "complete"},
                task_results=task_results,
                verdicts=analysis,
            )
    finally:
        ex.shutdown(wait=False)
        adapter.teardown() if hasattr(adapter, "teardown") else None

    if coverage is not None:
        for line in coverage.as_lines():
            logger.info("%s", line)

    return RunSummary(run_id=run_id, total=len(specs), passed=passed,
                      records=len(ctx.records), db_path=str(getattr(store, "db_path", "?")),
                      coverage=coverage, execution_failures=execution_failures,
                      oracle_failures=oracle_failures)


def _maybe_proxy(cfg: RunConfig):
    """Return a context manager: the ReplayProxy when record/replay, else a
    null context yielding None."""
    if cfg.replay_mode == "off":
        from contextlib import nullcontext
        return nullcontext(None)
    from src.replay import ReplayProxy
    if cfg.replay_mode == "replay":
        from src.replay import validate_fixture
        ok, issues, _ = validate_fixture(cfg.fixture)
        if not ok:
            raise ValueError(f"fixture failed validation: {issues}")
        return ReplayProxy(
            mode="replay",
            fixture=cfg.fixture,
            port=cfg.proxy_port or 4001,
        )
    # record: forward to the real LLM (cfg.upstream or $OPENAI_BASE_URL). The
    # proxy inherits OPENAI_API_KEY from this process env and re-signs the
    # Authorization header (the agent only has the proxy's dummy key).
    upstream = cfg.upstream or os.environ.get("OPENAI_BASE_URL")
    if not upstream:
        raise ValueError(
            "record mode needs an upstream LLM URL — set --upstream or $OPENAI_BASE_URL"
        )
    return ReplayProxy(
        mode="record", fixture=cfg.fixture, upstream=upstream,
        upstream_key=os.environ.get("OPENAI_API_KEY"),
        port=cfg.proxy_port or 4001,
    )


__all__ = ["RunConfig", "RunCoverage", "RunSummary", "run_benchmark"]
