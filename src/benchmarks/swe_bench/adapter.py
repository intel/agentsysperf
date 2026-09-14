#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""
SWE-Bench Benchmark Adapter for AgentSysPerf
==========================================

Drives SWE-Bench tasks through mini-swe-agent with latency tracking.

Follows the upstream mini-swe-agent runner pattern:
  - Uses HuggingFace datasets to load SWE-bench instances
  - Runs mini-swe-agent with Docker environments per instance
  - Records per-step LLM inference + tool execution latencies
  - Captures git diff patches as submissions

When mini-swe-agent/datasets are unavailable, falls back to sample tasks
for testing the adapter protocol.

Reference: https://github.com/princeton-nlp/SWE-bench
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from src.protocols import (
    AgentInvoker,
    TaskResult,
    TaskSpec,
)

# Scratch root for run artifacts. gettempdir() honours TMPDIR and falls
# back to "/tmp", so this resolves to the same paths as the hardcoded
# /tmp literals it replaced while no longer pinning output to a
# world-writable directory on systems that configure TMPDIR elsewhere.
_TMP = tempfile.gettempdir()

logger = logging.getLogger(__name__)

# Neither mini-swe-agent nor the SWE-bench run configs are vendored here, so both
# locations are properties of the host. Point each at its own checkout; they are
# independent upstream projects and need not share a parent directory.
MINI_SWE_AGENT_SRC = Path(
    os.environ.get("AGENTSYSPERF_MINI_SWE_AGENT_SRC", "~/mini-swe-agent/src")
).expanduser()
SWE_BENCH_DIR = Path(
    os.environ.get("AGENTSYSPERF_SWE_BENCH_DIR", "~/swe-bench-verified")
).expanduser()

DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
}

# Which Hub revision of the dataset to load. An unpinned load_dataset() takes
# whatever HEAD of the dataset repo is today, which is both a supply-chain hole
# (a mutated upstream silently changes what we execute) and a reproducibility
# hole — two runs a month apart can score different instance sets and the
# comparison looks like a regression. "main" is what an unpinned call already
# resolves to, so this default changes nothing; set AGENTSYSPERF_SWE_BENCH_REVISION
# to a commit SHA to make a measurement campaign byte-reproducible.
DATASET_REVISION = os.environ.get("AGENTSYSPERF_SWE_BENCH_REVISION", "main")

DEFAULT_CONFIG_TOOLCALL = SWE_BENCH_DIR / "vllm_swebench.yaml"
DEFAULT_CONFIG_TEXTBASED = SWE_BENCH_DIR / "vllm_swebench_textbased.yaml"

_LATENCY_LOCK = threading.Lock()
_PREDS_LOCK = threading.Lock()


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
    """Thread-safe per-instance latency recorder."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}

    def start_instance(self, instance_id: str, model_name: str) -> None:
        with _LATENCY_LOCK:
            self._data[instance_id] = {
                "instance_id": instance_id,
                "model_name": model_name,
                "start_time": time.time(),
                "end_time": None,
                "wall_time_s": None,
                "exit_status": None,
                "num_llm_calls": 0,
                "num_tool_calls": 0,
                "llm_latencies_s": [],
                "tool_latencies_s": [],
                "llm_calls": [],
                "tool_calls": [],
            }

    def end_instance(self, instance_id: str, exit_status: str | None) -> None:
        with _LATENCY_LOCK:
            entry = self._data.get(instance_id)
            if entry is None:
                return
            entry["end_time"] = time.time()
            entry["wall_time_s"] = entry["end_time"] - entry["start_time"]
            entry["exit_status"] = exit_status

    def record_llm(self, instance_id: str, record: dict) -> None:
        with _LATENCY_LOCK:
            entry = self._data.get(instance_id)
            if entry is None:
                return
            entry["num_llm_calls"] += 1
            if record.get("latency_s") is not None:
                entry["llm_latencies_s"].append(record["latency_s"])
            entry["llm_calls"].append(record)

    def record_tool(self, instance_id: str, record: dict) -> None:
        with _LATENCY_LOCK:
            entry = self._data.get(instance_id)
            if entry is None:
                return
            entry["num_tool_calls"] += 1
            if record.get("latency_s") is not None:
                entry["tool_latencies_s"].append(record["latency_s"])
            entry["tool_calls"].append(record)

    def get_instance(self, instance_id: str) -> Optional[dict]:
        with _LATENCY_LOCK:
            return self._data.get(instance_id)

    def summary(self) -> dict[str, Any]:
        with _LATENCY_LOCK:
            all_llm: list[float] = []
            all_tool: list[float] = []
            for entry in self._data.values():
                all_llm.extend(entry["llm_latencies_s"])
                all_tool.extend(entry["tool_latencies_s"])
            return {
                "num_instances": len(self._data),
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
                "instances": self._data,
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snapshot, indent=2, default=str))


def _try_import_minisweagent():
    """Try to import minisweagent; add $AGENTSYSPERF_MINI_SWE_AGENT_SRC to path if needed."""
    if str(MINI_SWE_AGENT_SRC) not in sys.path:
        sys.path.insert(0, str(MINI_SWE_AGENT_SRC))
    try:
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.config import get_config_path  # noqa: F401
        from minisweagent.environments import get_environment_class
        from minisweagent.models.litellm_model import LitellmModel
        return True
    except ImportError:
        return False


def _load_config_from_file(config_file) -> dict:
    """Load a mini-swe-agent YAML config.

    mini-swe-agent 1.13.0 removed ``get_config_from_spec``; ``get_config_path``
    resolves the spec to a path, which we then parse with yaml.
    """
    import yaml
    from minisweagent.config import get_config_path
    path = get_config_path(str(config_file))
    return yaml.safe_load(path.read_text())


def _try_import_datasets():
    """Try to import HuggingFace datasets."""
    try:
        from datasets import load_dataset
        return True
    except ImportError:
        return False


def _get_docker_image_name(instance: dict) -> str:
    """Get Docker image name for a SWE-bench instance."""
    image_name = instance.get("image_name")
    if image_name is None:
        iid = instance["instance_id"]
        id_docker = iid.replace("__", "_1776_")
        image_name = f"docker.io/swebench/sweb.eval.x86_64.{id_docker}:latest".lower()
    return image_name


class SWEBenchAdapter:
    """BenchmarkAdapter for SWE-Bench benchmark.

    When mini-swe-agent + datasets are available, drives the real benchmark
    with Docker environments and full latency instrumentation (matches
    the upstream swe-bench run_benchmark.py).

    When dependencies are unavailable, provides sample tasks for protocol testing.

    Parameters
    ----------
    subset : str
        SWE-bench subset: "lite", "verified", "full".
    split : str
        Dataset split: "dev", "test".
    model_name : str
        LiteLLM model string.
    vllm_base_url : str
        vLLM endpoint.
    text_based : bool
        Use text-based (code-fence) model instead of tool-calls.
    workers : int
        Concurrent worker threads.
    output_dir : Path, optional
        Directory for results, trajectories, and latency JSON.
    disable_thinking : bool
        Disable thinking mode for models that support it.
    step_limit : int, optional
        Max agent steps per instance.
    slice_spec : str
        Slice for dataset (e.g. ":5" for first 5).
    """

    name = "swe_bench"
    version = "2.0.0"

    def __init__(
        self,
        *,
        subset: str = "lite",
        split: str = "dev",
        model_name: str = "hosted_vllm/Qwen/Qwen2.5-Coder-7B-Instruct",
        vllm_base_url: str = "http://localhost:8000/v1",
        vllm_api_key: str = "EMPTY",
        text_based: bool = False,
        workers: int = 1,
        output_dir: Optional[Path] = None,
        disable_thinking: bool = False,
        step_limit: Optional[int] = None,
        cost_limit: Optional[float] = None,
        slice_spec: str = "",
        filter_spec: str = "",
        shuffle: bool = False,
        resume: bool = False,
    ) -> None:
        self._subset = subset
        self._split = split
        self._model_name = model_name
        self._vllm_base_url = vllm_base_url
        self._vllm_api_key = vllm_api_key
        self._text_based = text_based
        self._workers = workers
        self._output_dir = output_dir or Path(f"{_TMP}/agentsysperf_swe_bench")
        self._disable_thinking = disable_thinking
        self._step_limit = step_limit
        self._cost_limit = cost_limit
        self._slice_spec = slice_spec
        self._filter_spec = filter_spec
        self._shuffle = shuffle
        self._resume = resume

        self._mswa_available = _try_import_minisweagent()
        self._datasets_available = _try_import_datasets()
        self._recorder = LatencyRecorder()
        self._instances_cache: Optional[list[dict]] = None

    def list_tasks(
        self,
        *,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> Iterable[TaskSpec]:
        """Enumerate SWE-Bench instances from HuggingFace datasets."""
        instances = self._load_instances()

        count = 0
        for instance in instances:
            instance_id = instance["instance_id"]
            repo = instance.get("repo", "")

            if include and not any(inc in instance_id or inc in repo for inc in include):
                continue
            if exclude and any(exc in instance_id or exc in repo for exc in exclude):
                continue

            yield TaskSpec(
                id=instance_id,
                instruction=instance.get("problem_statement", ""),
                category=repo,
                difficulty="medium",
                timeout_s=1800.0,
                extra={
                    "repo": repo,
                    "instance_id": instance_id,
                    "base_commit": instance.get("base_commit", ""),
                    "hints_text": instance.get("hints_text", ""),
                    "patch": instance.get("patch", ""),
                    "test_patch": instance.get("test_patch", ""),
                    "pass_to_pass": instance.get("PASS_TO_PASS", instance.get("pass_to_pass", [])),
                    "fail_to_pass": instance.get("FAIL_TO_PASS", instance.get("fail_to_pass", [])),
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
        """Run one SWE-Bench instance with mini-swe-agent + Docker.

        Uses mini-swe-agent directly (same as the upstream swe-bench
        run_benchmark.py). The agent_invoker is NOT used because mini-swe-agent
        manages its own LLM interaction loop — AgentSysPerf measurements are captured
        via the timed model/environment wrappers.
        """
        instance_id = task.extra["instance_id"]
        logger.info(f"Running SWE-Bench instance {instance_id}")

        if not self._mswa_available:
            return self._run_task_fallback(task, agent_invoker=agent_invoker)

        try:
            return self._run_mswa_instance(task)
        except Exception as e:
            logger.error(f"Instance {instance_id} failed: {e}")
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
            logger.info(f"SWE-Bench latencies written to {latency_path}")
        self._instances_cache = None

    # ─── Internal: mini-swe-agent integration ───────────────────────

    def _load_instances(self) -> list[dict]:
        """Load instances from HuggingFace or return samples."""
        if self._instances_cache is not None:
            return self._instances_cache

        if self._datasets_available:
            try:
                from datasets import load_dataset
                import re
                import random

                dataset_path = DATASET_MAPPING.get(self._subset, self._subset)
                logger.info(
                    f"Loading dataset {dataset_path} split={self._split} "
                    f"revision={DATASET_REVISION}"
                )
                instances = list(load_dataset(
                    dataset_path, split=self._split, revision=DATASET_REVISION
                ))

                if self._shuffle:
                    instances = sorted(instances, key=lambda x: x["instance_id"])
                    random.seed(42)
                    random.shuffle(instances)
                if self._filter_spec:
                    instances = [i for i in instances if re.match(self._filter_spec, i["instance_id"])]
                if self._slice_spec:
                    parts = [int(x) if x else None for x in self._slice_spec.split(":")]
                    instances = instances[slice(*parts)]

                self._instances_cache = instances
                logger.info(f"Loaded {len(instances)} SWE-bench instances")
                return self._instances_cache
            except Exception as e:
                logger.warning(f"Could not load SWE-bench dataset: {e}")

        self._instances_cache = self._generate_sample_instances()
        return self._instances_cache

    def _run_mswa_instance(self, task: TaskSpec) -> TaskResult:
        """Run a single instance through mini-swe-agent."""
        if str(MINI_SWE_AGENT_SRC) not in sys.path:
            sys.path.insert(0, str(MINI_SWE_AGENT_SRC))

        from minisweagent.agents.default import DefaultAgent
        from minisweagent.environments import get_environment_class
        from minisweagent.models.litellm_model import LitellmModel

        try:
            from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel
        except ImportError:
            LitellmTextbasedModel = None

        instance_id = task.extra["instance_id"]
        instance = task.extra.get("raw_instance", task.extra)

        config_file = DEFAULT_CONFIG_TEXTBASED if self._text_based else DEFAULT_CONFIG_TOOLCALL
        if config_file.exists():
            config = _load_config_from_file(config_file)
        else:
            config = self._build_default_config()

        config["_text_based"] = self._text_based
        model_cfg = config.setdefault("model", {})
        model_kwargs = model_cfg.setdefault("model_kwargs", {})
        model_kwargs["api_base"] = self._vllm_base_url
        model_kwargs.setdefault("api_key", self._vllm_api_key)
        if self._model_name:
            model_cfg["model_name"] = self._model_name
        if self._disable_thinking:
            model_kwargs.setdefault("extra_body", {})
            model_kwargs["extra_body"]["chat_template_kwargs"] = {"enable_thinking": False}
        if self._step_limit is not None:
            config.setdefault("agent", {})["step_limit"] = self._step_limit
        if self._cost_limit is not None:
            config.setdefault("agent", {})["cost_limit"] = self._cost_limit

        self._recorder.start_instance(instance_id, self._model_name)

        exit_status: str | None = None
        submission: str = ""
        agent: DefaultAgent | None = None

        try:
            model_cfg_copy = dict(config.get("model", {}))
            model_cfg_copy.pop("model_class", None)
            # mini-swe-agent 1.13.0's LitellmModelConfig only accepts
            # model_name/model_kwargs/litellm_model_registry. Older configs
            # carry template fields (observation_template, format_error_template)
            # in the model block; those now belong to the agent, so move any
            # unknown keys out before constructing the model.
            import dataclasses
            from minisweagent.models.litellm_model import LitellmModelConfig
            _allowed = {f.name for f in dataclasses.fields(LitellmModelConfig)}
            _model_extras = {k: model_cfg_copy.pop(k) for k in list(model_cfg_copy)
                             if k not in _allowed}

            ModelCls = LitellmModel
            if self._text_based and LitellmTextbasedModel is not None:
                ModelCls = LitellmTextbasedModel

            # Local/self-hosted models aren't in litellm's price map, so its
            # cost calculator raises and mini-swe re-raises out of query().
            # Register a zero-cost entry for our model so cost calc succeeds.
            _reg_name = model_cfg_copy.get("model_name")
            if _reg_name:
                try:
                    import litellm
                    # Prices, not passwords — bandit's B105 matches the
                    # `*_token` key name and cannot tell the difference.
                    _zero = {
                        # price, not a secret
                        "input_cost_per_token": 0.0,  # nosec B105
                        # price, not a secret
                        "output_cost_per_token": 0.0,  # nosec B105
                        "litellm_provider": "openai",
                        "mode": "chat",
                    }
                    litellm.register_model({
                        _reg_name: _zero,
                        _reg_name.split("/", 1)[-1]: _zero,
                    })
                except Exception:
                    pass

            timed_cls = _make_timed_model_class(ModelCls, self._recorder, instance_id)
            model = timed_cls(**model_cfg_copy)

            env_config = dict(config.get("environment", {}))
            env_config.setdefault("environment_class", "docker")
            image_name = _get_docker_image_name({"instance_id": instance_id})
            if env_config["environment_class"] in ("docker", "swerex_modal"):
                env_config["image"] = image_name
            elif env_config["environment_class"] in ("singularity", "contree"):
                env_config["image"] = "docker://" + image_name

            base_env_cls = get_environment_class(env_config.pop("environment_class"))
            timed_env_cls = _make_timed_environment(base_env_cls, self._recorder, instance_id)
            # Filter to the fields the environment's config dataclass accepts —
            # older configs kept keys (e.g. 'interpreter') that the 1.13.0
            # environment dataclasses no longer define. The config class is the
            # default of the env's `config_class` __init__ kwarg.
            import inspect
            _sig = inspect.signature(base_env_cls.__init__)
            _env_cfg_cls = _sig.parameters.get("config_class")
            _env_cfg_cls = _env_cfg_cls.default if _env_cfg_cls is not None else None
            if _env_cfg_cls is not None and dataclasses.is_dataclass(_env_cfg_cls):
                _env_allowed = {f.name for f in dataclasses.fields(_env_cfg_cls)}
                env_config = {k: v for k, v in env_config.items() if k in _env_allowed}
            env = timed_env_cls(**env_config)

            startup_command = config.get("run", {}).get("env_startup_command")
            if startup_command:
                from jinja2 import StrictUndefined, Template
                rendered = Template(startup_command, undefined=StrictUndefined).render(**instance)
                out = env.execute({"command": rendered})
                if out["returncode"] != 0:
                    raise RuntimeError(f"Startup command failed: {out}")

            agent_kwargs = dict(config.get("agent", {}))
            agent_kwargs.pop("agent_class", None)
            agent_kwargs.pop("mode", None)
            agent_kwargs.pop("confirm_exit", None)
            # Fold template fields that older configs kept in the model block
            # into the agent, then filter to the fields AgentConfig accepts so
            # renamed/removed keys (e.g. observation_template) don't error.
            agent_kwargs.update(_model_extras)
            from minisweagent.agents.default import AgentConfig as _AgentConfig
            _agent_allowed = {f.name for f in dataclasses.fields(_AgentConfig)}
            agent_kwargs = {k: v for k, v in agent_kwargs.items() if k in _agent_allowed}
            # mini-swe 1.13.0's format_error_template is rendered with `actions`,
            # not `error`; older configs use `{{error}}`, which StrictUndefined
            # rejects. Rewrite the reference so the format-error path renders.
            _fet = agent_kwargs.get("format_error_template")
            if isinstance(_fet, str) and "error" in _fet:
                import re as _re
                agent_kwargs["format_error_template"] = _re.sub(
                    r"\{\{\s*error\s*\}\}", "{{actions|length}} action(s) found", _fet)
            agent = DefaultAgent(model, env, **agent_kwargs)

            problem_statement = task.instruction
            # 1.13.0's DefaultAgent.run returns (exit_status, submission);
            # older versions returned a dict.
            info = agent.run(problem_statement)
            if isinstance(info, tuple):
                exit_status, submission = info[0], info[1]
            else:
                exit_status = info.get("exit_status")
                submission = info.get("submission", "")

        except Exception as e:
            logger.error(f"[{instance_id}] error: {e}", exc_info=True)
            exit_status = type(e).__name__
        finally:
            self._recorder.end_instance(instance_id, exit_status)

            instance_dir = self._output_dir / instance_id
            instance_dir.mkdir(parents=True, exist_ok=True)
            if agent is not None:
                traj_path = instance_dir / f"{instance_id}.traj.json"
                try:
                    agent.save(traj_path, {
                        "info": {"exit_status": exit_status, "submission": submission},
                        "instance_id": instance_id,
                    })
                except Exception:
                    pass

            preds_path = self._output_dir / "preds.json"
            _update_preds(preds_path, instance_id, self._model_name, submission)

            try:
                if hasattr(env, "cleanup"):
                    env.cleanup()
            except Exception:
                pass

        latency_data = self._recorder.get_instance(instance_id)
        passed = exit_status in (None, "ok", "submitted")
        has_patch = bool(submission and len(submission) > 10)

        extra: dict[str, Any] = {
            "exit_status": exit_status,
            "has_patch": has_patch,
            "patch_length": len(submission) if submission else 0,
        }
        if latency_data:
            extra.update({
                "wall_time_s": latency_data.get("wall_time_s"),
                "num_llm_calls": latency_data.get("num_llm_calls", 0),
                "num_tool_calls": latency_data.get("num_tool_calls", 0),
                "llm_latency_stats": _stats(latency_data.get("llm_latencies_s", [])),
                "tool_latency_stats": _stats(latency_data.get("tool_latencies_s", [])),
            })

        reward = 0.0
        if has_patch:
            reward = 0.5
        if passed and has_patch:
            reward = 1.0

        return TaskResult(
            task_id=task.id,
            passed=passed and has_patch,
            reward=reward,
            error=None if passed else exit_status,
            extra=extra,
        )

    def _build_default_config(self) -> dict:
        """Build a default config when no YAML file is available."""
        return {
            "agent": {
                "system_template": (
                    "You are a helpful assistant that can interact with a computer "
                    "shell to solve programming tasks."
                ),
                "step_limit": self._step_limit or 250,
            },
            "environment": {
                "cwd": "/testbed",
                "timeout": 60,
                "environment_class": "docker",
            },
            "model": {
                "model_name": self._model_name,
                "cost_tracking": "ignore_errors",
                "model_kwargs": {
                    "api_base": self._vllm_base_url,
                    "api_key": self._vllm_api_key,
                    "temperature": 0.0,
                    "max_completion_tokens": 4096,
                    "timeout": 180,
                },
            },
        }

    # ─── Fallback: sample tasks when dependencies unavailable ───────

    def _run_task_fallback(
        self, task: TaskSpec, *, agent_invoker: AgentInvoker
    ) -> TaskResult:
        """Run task using AgentInvoker when mini-swe-agent is not installed."""
        logger.warning("mini-swe-agent not available; running in fallback mode via agent_invoker")

        t0 = time.time()
        try:
            response = agent_invoker.invoke(
                instruction=task.instruction,
                environment={"repo": task.extra.get("repo", "")},
                metadata={"task_type": "code_fix", "repo": task.extra.get("repo", "")},
                session_hint=task.id,
            )

            patch = ""
            if isinstance(response, str):
                patch = response
            elif isinstance(response, dict):
                patch = response.get("patch", response.get("submission", ""))

            wall_time = time.time() - t0
            has_patch = bool(patch and len(patch) > 10)

            return TaskResult(
                task_id=task.id,
                passed=has_patch,
                reward=0.5 if has_patch else 0.0,
                extra={"wall_time_s": wall_time, "mode": "fallback", "has_patch": has_patch},
            )
        except Exception as e:
            return TaskResult(
                task_id=task.id,
                passed=False,
                reward=0.0,
                error=str(e),
            )

    def _generate_sample_instances(self) -> list[dict]:
        """Sample instances for protocol testing."""
        return [
            {
                "instance_id": "django__django-11583",
                "repo": "django/django",
                "base_commit": "abc123",
                "problem_statement": (
                    "Bug: Auto-created intermediate models for M2M fields don't get "
                    "the same db_tablespace as the model.\n\n"
                    "When a model with db_tablespace set has a ManyToManyField, "
                    "the auto-created intermediate model should inherit the tablespace."
                ),
                "hints_text": "Check django/db/models/fields/related.py",
                "patch": "",
                "test_patch": "",
                "PASS_TO_PASS": [],
                "FAIL_TO_PASS": ["tests.model_fields.test_manytomanyfield.ManyToManyFieldTests.test_m2m_db_tablespace"],
            },
            {
                "instance_id": "scikit-learn__scikit-learn-13779",
                "repo": "scikit-learn/scikit-learn",
                "base_commit": "def456",
                "problem_statement": (
                    "VotingClassifier doesn't work with set_params when estimators "
                    "are set to 'drop'.\n\n"
                    "Setting an estimator to 'drop' in VotingClassifier and then "
                    "calling set_params raises an error."
                ),
                "hints_text": "Check sklearn/ensemble/voting.py",
                "patch": "",
                "test_patch": "",
                "PASS_TO_PASS": [],
                "FAIL_TO_PASS": ["sklearn/tests/test_voting.py::test_set_params_drop"],
            },
        ]


def _make_timed_model_class(base_cls: type, recorder: LatencyRecorder, instance_id: str):
    """Wrap a model class to record per-query latencies."""

    class TimedModel(base_cls):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._recorder = recorder
            self._instance_id = instance_id
            self._call_idx = 0
            self._last_response = None
            self._last_inference_latency_s: float | None = None

        def _query(self, messages, **kwargs):
            t0 = time.perf_counter()
            resp = super()._query(messages, **kwargs)
            self._last_inference_latency_s = time.perf_counter() - t0
            self._last_response = resp
            return resp

        def query(self, messages, **kwargs) -> dict:
            self._call_idx += 1
            self._last_response = None
            self._last_inference_latency_s = None
            t0 = time.perf_counter()
            start_ts = time.time()
            result: dict | None = None
            err_type: str | None = None
            try:
                result = super().query(messages, **kwargs)
                return result
            except Exception as e:
                err_type = type(e).__name__
                raise
            finally:
                total_latency = time.perf_counter() - t0
                usage = {"prompt_tokens": None, "completion_tokens": None}
                if self._last_response is not None:
                    try:
                        u_obj = getattr(self._last_response, "usage", None)
                        if u_obj and hasattr(u_obj, "model_dump"):
                            u = u_obj.model_dump()
                            usage = {
                                "prompt_tokens": u.get("prompt_tokens"),
                                "completion_tokens": u.get("completion_tokens"),
                            }
                    except Exception:
                        pass

                self._recorder.record_llm(
                    self._instance_id,
                    {
                        "call_idx": self._call_idx,
                        "start_time": start_ts,
                        "latency_s": total_latency,
                        "inference_latency_s": self._last_inference_latency_s,
                        "num_messages": len(messages),
                        "prompt_tokens": usage["prompt_tokens"],
                        "completion_tokens": usage["completion_tokens"],
                        "error_type": err_type,
                    },
                )

    TimedModel.__name__ = f"Timed{base_cls.__name__}"
    return TimedModel


def _make_timed_environment(base_cls: type, recorder: LatencyRecorder, instance_id: str):
    """Wrap an environment class to record per-execute latencies."""

    class TimedEnvironment(base_cls):
        def execute(self, action, *args, **kwargs):
            t0 = time.perf_counter()
            start_ts = time.time()
            err: str | None = None
            command = action.get("command", "") if isinstance(action, dict) else str(action)
            try:
                return super().execute(action, *args, **kwargs)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                raise
            finally:
                recorder.record_tool(
                    instance_id,
                    {
                        "start_time": start_ts,
                        "latency_s": time.perf_counter() - t0,
                        "command_preview": (command[:200] + "...") if len(command) > 200 else command,
                        "command_len": len(command),
                        "error": err,
                    },
                )

    TimedEnvironment.__name__ = f"Timed{base_cls.__name__}"
    return TimedEnvironment


def _update_preds(output_path: Path, instance_id: str, model_name: str, patch: str) -> None:
    """Update predictions JSON file."""
    with _PREDS_LOCK:
        data: dict[str, Any] = {}
        if output_path.exists():
            try:
                data = json.loads(output_path.read_text())
            except Exception:
                pass
        data[instance_id] = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": patch,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(data, indent=2))


__all__ = ["SWEBenchAdapter"]
