#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""
Tau-Bench Benchmark Adapter for AgentSysPerf
==========================================

Drives Tau-Bench (tau2) customer-service scenarios with latency tracking.

Follows the upstream tau-bench runner pattern:
  - Monkey-patches tau2 internals to capture per-simulation LLM + tool latencies
  - Connects to a local vLLM server via LiteLLM
  - Records per-task timing breakdown (LLM inference vs tool execution)

When tau2 is not installed, falls back to sample tasks for testing the
adapter protocol without a live vLLM server.

Reference: https://github.com/sierra-research/tau-bench
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from src.protocols import (
    AgentInvoker,
    TaskResult,
    TaskSpec,
)
import tempfile

# Scratch root for run artifacts. gettempdir() honours TMPDIR and falls
# back to "/tmp", so this resolves to the same paths as the hardcoded
# /tmp literals it replaced while no longer pinning output to a
# world-writable directory on systems that configure TMPDIR elsewhere.
_TMP = tempfile.gettempdir()

logger = logging.getLogger(__name__)

_LATENCY_LOCK = threading.Lock()
_current_sim: ContextVar[Optional[tuple[str, str]]] = ContextVar(
    "_tau2_bench_current_sim", default=None
)


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None, "p50": None, "p95": None, "sum": 0.0}
    sorted_v = sorted(values)
    n = len(sorted_v)
    def pct(p: float) -> float:
        k = max(0, min(n - 1, int(round((p / 100.0) * (n - 1)))))
        return sorted_v[k]
    return {
        "count": n,
        "mean": sum(sorted_v) / n,
        "min": sorted_v[0],
        "max": sorted_v[-1],
        "p50": pct(50),
        "p95": pct(95),
        "sum": sum(sorted_v),
    }


class LatencyRecorder:
    """Thread-safe per-simulation latency recorder."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}

    def _ensure(self, sim_key: str, task_id: str, simulation_id: str) -> dict[str, Any]:
        entry = self._data.get(sim_key)
        if entry is None:
            entry = {
                "sim_key": sim_key,
                "task_id": task_id,
                "simulation_id": simulation_id,
                "start_time": time.time(),
                "end_time": None,
                "wall_time_s": None,
                "exit_status": None,
                "num_llm_calls": 0,
                "num_tool_calls": 0,
                "num_agent_llm_calls": 0,
                "num_user_llm_calls": 0,
                "num_eval_llm_calls": 0,
                "llm_latencies_s": [],
                "tool_latencies_s": [],
                "llm_calls": [],
                "tool_calls": [],
            }
            self._data[sim_key] = entry
        return entry

    def start_sim(self, task_id: str, simulation_id: str) -> None:
        sim_key = f"{task_id}::{simulation_id}"
        with _LATENCY_LOCK:
            self._ensure(sim_key, task_id, simulation_id)

    def end_sim(self, task_id: str, simulation_id: str, exit_status: Optional[str]) -> None:
        sim_key = f"{task_id}::{simulation_id}"
        with _LATENCY_LOCK:
            entry = self._data.get(sim_key)
            if entry is None:
                return
            entry["end_time"] = time.time()
            entry["wall_time_s"] = entry["end_time"] - entry["start_time"]
            entry["exit_status"] = exit_status

    def record_llm(self, sim: tuple[str, str], record: dict) -> None:
        task_id, simulation_id = sim
        sim_key = f"{task_id}::{simulation_id}"
        with _LATENCY_LOCK:
            entry = self._ensure(sim_key, task_id, simulation_id)
            entry["num_llm_calls"] += 1
            cn = record.get("call_name") or ""
            if "user_simulator" in cn or cn in ("generate_user_message", "user_message"):
                entry["num_user_llm_calls"] += 1
            elif "nl_assertion" in cn or "eval" in cn or "review" in cn:
                entry["num_eval_llm_calls"] += 1
            else:
                entry["num_agent_llm_calls"] += 1
            if record.get("latency_s") is not None:
                entry["llm_latencies_s"].append(record["latency_s"])
            entry["llm_calls"].append(record)

    def record_tool(self, sim: tuple[str, str], record: dict) -> None:
        task_id, simulation_id = sim
        sim_key = f"{task_id}::{simulation_id}"
        with _LATENCY_LOCK:
            entry = self._ensure(sim_key, task_id, simulation_id)
            entry["num_tool_calls"] += 1
            if record.get("latency_s") is not None:
                entry["tool_latencies_s"].append(record["latency_s"])
            entry["tool_calls"].append(record)

    def get_sim_data(self, task_id: str, simulation_id: str) -> Optional[dict]:
        sim_key = f"{task_id}::{simulation_id}"
        with _LATENCY_LOCK:
            return self._data.get(sim_key)

    def summary(self) -> dict[str, Any]:
        with _LATENCY_LOCK:
            all_llm: list[float] = []
            all_tool: list[float] = []
            for entry in self._data.values():
                all_llm.extend(entry["llm_latencies_s"])
                all_tool.extend(entry["tool_latencies_s"])
            return {
                "num_simulations": len(self._data),
                "total_llm_calls": len(all_llm),
                "total_tool_calls": len(all_tool),
                "llm_latency_s": _stats(all_llm),
                "tool_latency_s": _stats(all_tool),
            }

    def dump(self, path: Path) -> None:
        with _LATENCY_LOCK:
            snapshot = {
                "generated_at": time.time(),
                "summary": self.summary(),
                "simulations": self._data,
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snapshot, indent=2, default=str))


def _try_import_tau2():
    """Try to import tau2; returns None if unavailable."""
    try:
        import tau2
        return tau2
    except ImportError:
        return None


def _install_tau2_patches(recorder: LatencyRecorder) -> bool:
    """Monkey-patch tau2 internals for latency recording.

    Returns True if patches were installed, False if tau2 is unavailable.
    Pattern from the upstream tau-bench runner.
    """
    try:
        from tau2.environment import environment as env_mod
        from tau2.runner import batch as batch_mod
        from tau2.utils import llm_utils
    except ImportError:
        return False

    orig_generate = llm_utils.generate
    orig_get_response = env_mod.Environment.get_response

    ctx_cls = (
        getattr(batch_mod, "_TaskLogContext", None)
        or getattr(batch_mod, "TaskRunCtx", None)
        or getattr(batch_mod, "TaskLogContext", None)
    )
    if ctx_cls is None:
        logger.warning("Could not find per-task context class in tau2.runner.batch")
        return False

    orig_enter = ctx_cls.__enter__
    orig_exit = ctx_cls.__exit__
    call_counter = {"idx": 0}

    def patched_enter(self):
        result = orig_enter(self)
        try:
            sim_key = (str(self.task.id), str(self.simulation_id))
        except Exception:
            sim_key = ("unknown", "unknown")
        _current_sim.set(sim_key)
        recorder.start_sim(*sim_key)
        return result

    def patched_exit(self, exc_type, exc_val, exc_tb):
        try:
            sim_key = (str(self.task.id), str(self.simulation_id))
            status = exc_type.__name__ if exc_type is not None else "ok"
            recorder.end_sim(*sim_key, status)
        except Exception:
            pass
        return orig_exit(self, exc_type, exc_val, exc_tb)

    ctx_cls.__enter__ = patched_enter
    ctx_cls.__exit__ = patched_exit

    def patched_generate(model, messages, tools=None, tool_choice=None,
                         call_name=None, **kwargs):
        sim = _current_sim.get()
        with _LATENCY_LOCK:
            call_counter["idx"] += 1
            call_idx = call_counter["idx"]

        t0 = time.perf_counter()
        start_ts = time.time()
        err_type: Optional[str] = None
        err_msg: Optional[str] = None
        response_message = None
        try:
            response_message = orig_generate(
                model, messages, tools=tools, tool_choice=tool_choice,
                call_name=call_name, **kwargs,
            )
            return response_message
        except Exception as e:
            err_type = type(e).__name__
            err_msg = str(e)
            raise
        finally:
            total_latency = time.perf_counter() - t0

            inference_latency = getattr(response_message, "generation_time_seconds", None)
            usage = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
            usage_obj = getattr(response_message, "usage", None)
            if usage_obj is not None:
                if hasattr(usage_obj, "model_dump"):
                    u = usage_obj.model_dump()
                elif isinstance(usage_obj, dict):
                    u = usage_obj
                else:
                    u = {}
                usage = {
                    "prompt_tokens": u.get("prompt_tokens"),
                    "completion_tokens": u.get("completion_tokens"),
                    "total_tokens": u.get("total_tokens"),
                }

            content = getattr(response_message, "content", None)
            raw_tool_calls = getattr(response_message, "tool_calls", None) or []
            cost = getattr(response_message, "cost", None)

            preview: Optional[str] = None
            if isinstance(content, str):
                preview = content[:300] + "…" if len(content) > 300 else content

            record = {
                "call_idx": call_idx,
                "start_time": start_ts,
                "latency_s": total_latency,
                "inference_latency_s": inference_latency,
                "call_name": call_name,
                "role": getattr(response_message, "role", None),
                "model": model,
                "num_messages": len(messages) if messages else None,
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
                "num_raw_tool_calls": len(raw_tool_calls),
                "cost": cost,
                "response_preview": preview,
                "error_type": err_type,
                "error": err_msg,
            }
            if sim is not None:
                recorder.record_llm(sim, record)

    llm_utils.generate = patched_generate
    patched_generate._orig = orig_generate  # type: ignore[attr-defined]
    _install_tau2_patches._patched_generate = patched_generate  # type: ignore[attr-defined]

    for mod_name, mod in list(sys.modules.items()):
        if not mod_name.startswith("tau2."):
            continue
        try:
            if getattr(mod, "generate", None) is orig_generate:
                mod.generate = patched_generate
        except Exception:
            pass

    def patched_get_response(self, message):
        sim = _current_sim.get()
        t0 = time.perf_counter()
        start_ts = time.time()
        err_type: Optional[str] = None
        err_msg: Optional[str] = None
        response = None
        try:
            response = orig_get_response(self, message)
            return response
        except Exception as e:
            err_type = type(e).__name__
            err_msg = str(e)
            raise
        finally:
            latency = time.perf_counter() - t0
            tool_name = getattr(message, "name", None)
            requestor = getattr(message, "requestor", None)
            args = getattr(message, "arguments", None)
            try:
                args_preview = json.dumps(args, default=str)[:300]
            except Exception:
                args_preview = repr(args)[:300] if args else None
            resp_error = bool(getattr(response, "error", False)) if response is not None else False
            resp_content = getattr(response, "content", None)
            resp_preview: Optional[str] = None
            if isinstance(resp_content, str):
                resp_preview = resp_content[:300] + "…" if len(resp_content) > 300 else resp_content
            record = {
                "start_time": start_ts,
                "latency_s": latency,
                "tool_name": tool_name,
                "requestor": requestor,
                "arguments_preview": args_preview,
                "response_error": resp_error or (err_type is not None),
                "response_preview": resp_preview,
                "error_type": err_type,
                "error": err_msg,
            }
            if sim is not None:
                recorder.record_tool(sim, record)

    env_mod.Environment.get_response = patched_get_response
    return True


def _rebind_generate_everywhere() -> None:
    """Re-walk sys.modules and rebind any unwrapped `generate` reference.

    Called after tau2's CLI imports complete, since agent/user modules pull in
    `generate` by name at import time.
    """
    patched = getattr(_install_tau2_patches, "_patched_generate", None)
    if patched is None:
        return
    orig = getattr(patched, "_orig", None)
    for mod_name, mod in list(sys.modules.items()):
        if not mod_name.startswith("tau2."):
            continue
        try:
            if getattr(mod, "generate", None) is orig:
                mod.generate = patched
        except Exception:
            pass


class TauBenchAdapter:
    """BenchmarkAdapter for Tau-Bench (tau2) benchmark.

    When tau2 + vLLM are available, drives the real benchmark with full latency
    instrumentation (matches the upstream tau-bench runner).

    When tau2 is unavailable, provides sample tasks for protocol testing.

    Parameters
    ----------
    domain : str
        tau2 domain: "retail", "airline", "telecom", "mock", "banking_knowledge".
    model_name : str
        LiteLLM model string (e.g. "hosted_vllm/Qwen/Qwen3-Coder-30B-A3B-Instruct").
    vllm_base_url : str
        vLLM OpenAI-compatible endpoint.
    num_trials : int
        Number of trials per task (simulations).
    max_steps : int
        Max agent steps per simulation.
    max_concurrency : int
        Concurrent simulations.
    output_dir : Path, optional
        Directory for latency JSON and tau2 results.
    task_split_name : str
        tau2 task split (default "base").
    temperature : float
        Sampling temperature.
    """

    name = "tau_bench"
    version = "2.0.0"

    def __init__(
        self,
        *,
        domain: str = "retail",
        model_name: str = "hosted_vllm/Qwen/Qwen3-Coder-30B-A3B-Instruct",
        vllm_base_url: str = "http://localhost:8000/v1",
        vllm_api_key: str = "EMPTY",
        num_trials: int = 1,
        max_steps: int = 100,
        max_concurrency: int = 1,
        output_dir: Optional[Path] = None,
        task_split_name: str = "base",
        temperature: float = 0.0,
        seed: int = 42,
    ) -> None:
        self._domain = domain
        self._model_name = model_name
        self._vllm_base_url = vllm_base_url
        self._vllm_api_key = vllm_api_key
        self._num_trials = num_trials
        self._max_steps = max_steps
        self._max_concurrency = max_concurrency
        self._output_dir = output_dir or Path(f"{_TMP}/agentsysperf_tau_bench")
        self._task_split_name = task_split_name
        self._temperature = temperature
        self._seed = seed

        self._tau2_available = _try_import_tau2() is not None
        self._recorder = LatencyRecorder()
        self._patches_installed = False
        self._tasks_cache: Optional[list[dict]] = None

    def list_tasks(
        self,
        *,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> Iterable[TaskSpec]:
        """Enumerate tau2 tasks from the specified domain and split."""
        tasks = self._load_tasks()

        count = 0
        for task_data in tasks:
            task_id = str(task_data.get("id", task_data.get("task_id", f"tau_{count}")))

            if include and not any(inc in task_id for inc in include):
                continue
            if exclude and any(exc in task_id for exc in exclude):
                continue

            yield TaskSpec(
                id=f"tau_bench/{self._domain}/{task_id}",
                instruction=task_data.get("user_instruction", task_data.get("instruction", "")),
                category=self._domain,
                difficulty="medium",
                timeout_s=900.0,
                extra={
                    "domain": self._domain,
                    "task_id": task_id,
                    "raw_task": task_data,
                },
            )

            count += 1
            if limit and count >= limit:
                break

    def run_task(
        self,
        task: TaskSpec,
        *,
        agent_invoker: AgentInvoker,
        on_step: Optional[Any] = None,
    ) -> TaskResult:
        """Run one tau2 simulation with full latency instrumentation.

        Uses the tau2 library directly (same as the upstream tau-bench
        run_tau_benchmark.py). The agent_invoker is NOT used because tau2
        manages its own LLM calls internally — AgentSysPerf measurements are
        captured via the monkey-patched latency recorder.
        """
        task_id = task.extra["task_id"]
        logger.info(f"Running tau2 task {task_id} (domain: {self._domain})")

        if not self._tau2_available:
            return self._run_task_fallback(task, agent_invoker=agent_invoker)

        if not self._patches_installed:
            self._setup_tau2_env()
            self._patches_installed = _install_tau2_patches(self._recorder)

        try:
            return self._run_tau2_task(task)
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            return TaskResult(
                task_id=task.id,
                passed=False,
                reward=0.0,
                error=str(e),
            )

    def teardown(self) -> None:
        """Dump latency data and clean up."""
        if self._recorder._data:
            latency_path = self._output_dir / "latencies.json"
            self._recorder.dump(latency_path)
            logger.info(f"Tau-Bench latencies written to {latency_path}")
        self._tasks_cache = None

    # ─── Internal: tau2 integration ─────────────────────────────────

    def _setup_tau2_env(self) -> None:
        """Configure environment variables for tau2 + vLLM."""
        os.environ.setdefault("HOSTED_VLLM_API_BASE", self._vllm_base_url)
        os.environ.setdefault("HOSTED_VLLM_API_KEY", self._vllm_api_key)
        os.environ.setdefault("OPENAI_API_KEY", self._vllm_api_key)
        os.environ.setdefault("OPENAI_API_BASE", self._vllm_base_url)

        for _var in ("no_proxy", "NO_PROXY"):
            _existing = os.environ.get(_var, "")
            _localhost = "localhost,127.0.0.1"
            if _localhost not in _existing:
                os.environ[_var] = f"{_existing},{_localhost}" if _existing else _localhost

        try:
            import litellm  # noqa
            from tau2.utils import llm_utils as _llm_utils
            _llm_utils.get_response_cost = lambda response: 0.0
        except Exception:
            pass

        eval_model = self._model_name
        eval_args = {
            "temperature": self._temperature,
            "api_base": self._vllm_base_url,
            "api_key": self._vllm_api_key,
        }
        try:
            from tau2 import config as _tau2_cfg
            _tau2_cfg.DEFAULT_LLM_NL_ASSERTIONS = eval_model
            _tau2_cfg.DEFAULT_LLM_NL_ASSERTIONS_ARGS = eval_args
            _tau2_cfg.DEFAULT_LLM_ENV_INTERFACE = eval_model
            _tau2_cfg.DEFAULT_LLM_ENV_INTERFACE_ARGS = eval_args
        except Exception:
            pass

    def _load_tasks(self) -> list[dict]:
        """Load tasks from tau2 domain or return sample tasks."""
        if self._tasks_cache is not None:
            return self._tasks_cache

        if self._tau2_available:
            try:
                import tau2.registry as reg
                loader = reg.get_tasks_loader(self._domain)
                raw_tasks = loader()
                self._tasks_cache = []
                for t in raw_tasks:
                    d = t.model_dump()
                    scenario = d.get("user_scenario") or {}
                    instructions = scenario.get("instructions") or {}
                    reason = instructions.get("reason_for_call", "")
                    self._tasks_cache.append({
                        "id": str(d["id"]),
                        "user_instruction": reason,
                        "domain": instructions.get("domain", self._domain),
                    })
                logger.info(f"Loaded {len(self._tasks_cache)} tasks from tau2 domain '{self._domain}'")
                return self._tasks_cache
            except Exception as e:
                logger.warning(f"Could not load tau2 tasks: {e}")

        self._tasks_cache = self._generate_sample_tasks()
        return self._tasks_cache

    def _run_tau2_task(self, task: TaskSpec) -> TaskResult:
        """Run a single task through tau2's runner."""
        from tau2.cli import main as tau2_main

        task_id = task.extra["task_id"]
        save_to = f"agentsysperf_{self._domain}_{task_id}"
        llm_args = json.dumps({
            "temperature": self._temperature,
            "api_base": self._vllm_base_url,
            "api_key": self._vllm_api_key,
        })

        cli_argv = [
            "tau2", "run",
            "--domain", self._domain,
            "--agent-llm", self._model_name,
            "--user-llm", self._model_name,
            "--agent-llm-args", llm_args,
            "--user-llm-args", llm_args,
            "--num-trials", str(self._num_trials),
            "--max-steps", str(self._max_steps),
            "--max-concurrency", str(self._max_concurrency),
            "--seed", str(self._seed),
            "--log-level", "WARNING",
            "--save-to", save_to,
            "--task-split-name", self._task_split_name,
            "--task-ids", task_id,
            "--auto-resume",
        ]

        _rebind_generate_everywhere()

        orig_argv = sys.argv
        sys.argv = cli_argv
        exit_status = "ok"
        try:
            tau2_main()
        except SystemExit as e:
            if e.code not in (None, 0):
                exit_status = f"exit_{e.code}"
        except Exception as e:
            exit_status = type(e).__name__
            logger.error(f"tau2 run failed for task {task_id}: {e}")
        finally:
            sys.argv = orig_argv

        sim_data = self._recorder.get_sim_data(task_id, "0")
        if sim_data is None:
            for key, data in self._recorder._data.items():
                if task_id in key:
                    sim_data = data
                    break

        passed = exit_status == "ok"
        reward = 1.0 if passed else 0.0

        extra: dict[str, Any] = {"exit_status": exit_status}
        if sim_data:
            extra.update({
                "start_time": sim_data.get("start_time"),
                "wall_time_s": sim_data.get("wall_time_s"),
                "num_llm_calls": sim_data.get("num_llm_calls", 0),
                "num_tool_calls": sim_data.get("num_tool_calls", 0),
                "num_agent_llm_calls": sim_data.get("num_agent_llm_calls", 0),
                "num_user_llm_calls": sim_data.get("num_user_llm_calls", 0),
                "num_eval_llm_calls": sim_data.get("num_eval_llm_calls", 0),
                "llm_latency_stats": _stats(sim_data.get("llm_latencies_s", [])),
                "tool_latency_stats": _stats(sim_data.get("tool_latencies_s", [])),
                "llm_calls": sim_data.get("llm_calls", []),
                "tool_calls": sim_data.get("tool_calls", []),
            })

        return TaskResult(
            task_id=task.id,
            passed=passed,
            reward=reward,
            error=None if passed else exit_status,
            extra=extra,
        )

    # ─── Fallback: sample tasks when tau2 is unavailable ────────────

    def _run_task_fallback(
        self, task: TaskSpec, *, agent_invoker: AgentInvoker
    ) -> TaskResult:
        """Run task using AgentInvoker when tau2 is not installed."""
        logger.warning("tau2 not available; running in fallback mode via agent_invoker")

        prompt = f"Customer Service ({self._domain}): {task.instruction}"
        t0 = time.time()
        try:
            response = agent_invoker.invoke(
                instruction=prompt,
                metadata={"domain": self._domain, "task_type": "customer_service"},
                session_hint=task.id,
            )
            wall_time = time.time() - t0
            return TaskResult(
                task_id=task.id,
                passed=True,
                reward=0.5,
                extra={"wall_time_s": wall_time, "mode": "fallback"},
            )
        except Exception as e:
            return TaskResult(
                task_id=task.id,
                passed=False,
                reward=0.0,
                error=str(e),
            )

    def _generate_sample_tasks(self) -> list[dict]:
        """Sample tasks for protocol testing."""
        return [
            {
                "id": "sample_retail_001",
                "user_instruction": (
                    "I'm looking for a blue cotton t-shirt in size medium, "
                    "preferably under $30. Do you have anything in stock?"
                ),
            },
            {
                "id": "sample_retail_002",
                "user_instruction": (
                    "I bought a laptop last week but it's defective. "
                    "I'd like to return it for a refund. Order ORD-12345."
                ),
            },
            {
                "id": "sample_airline_001",
                "user_instruction": (
                    "I need to book a round-trip flight from New York to London "
                    "departing June 15th, returning June 22nd. Budget $1200."
                ),
            },
        ]


__all__ = ["TauBenchAdapter"]
