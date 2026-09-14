#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""
vLLM CPU Inference Benchmark Adapter for AgentSysPerf
====================================================

Benchmarks vLLM inference serving on CPU with phase-aware measurement
(prefill vs decode). Designed to produce the data signatures that
MemoryBandwidthAnalyzer uses to identify speculative decoding and
other memory-bound optimization opportunities.

Separates prefill and decode measurement because they have fundamentally
different hardware profiles:
- Prefill: compute-bound (GEMM on prompt tokens), high IPC, AMX-friendly
- Decode: memory-bandwidth-bound (weight streaming per token), low IPC

Tuning reference: vLLM's CPU installation and performance guidance,
https://docs.vllm.ai/en/latest/getting_started/installation/cpu/

Prerequisites:
- vLLM installed or Docker image available
- Model weights accessible (HuggingFace cache or local path)
- Sufficient RAM for model + KV cache (model_size + VLLM_CPU_KVCACHE_SPACE)

Usage:
    adapter = VLLMInferenceAdapter(
        model="meta-llama/Llama-3.2-1B-Instruct",
        mode="docker",
    )
    for task in adapter.list_tasks():
        result = adapter.run_task(task, agent_invoker=invoker)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.protocols import (
    AgentInvoker,
    TaskResult,
    TaskSpec,
)

logger = logging.getLogger(__name__)


# ─── Task Definitions ────────────────────────────────────────────────────

# Each task exercises a specific phase/pattern of LLM inference
INFERENCE_TASKS = [
    {
        "id": "decode_batch1_short",
        "instruction": "Measure autoregressive decode at batch=1, short output",
        "category": "decode",
        "difficulty": "baseline",
        "extra": {
            "phase": "decode",
            "batch_size": 1,
            "prompt_tokens": 128,
            "max_output_tokens": 64,
            "num_requests": 10,
            "description": (
                "Single-request decode: maximum memory-bandwidth pressure. "
                "Expected: IPC < 1.5, cache miss > 80%, weight_streaming pattern."
            ),
        },
    },
    {
        "id": "decode_batch1_long",
        "instruction": "Measure autoregressive decode at batch=1, long output",
        "category": "decode",
        "difficulty": "baseline",
        "extra": {
            "phase": "decode",
            "batch_size": 1,
            "prompt_tokens": 128,
            "max_output_tokens": 512,
            "num_requests": 5,
            "description": (
                "Long-generation decode: sustained weight streaming. "
                "KV-cache grows throughout generation. Watch for kv_cache_pressure."
            ),
        },
    },
    {
        "id": "decode_batch8",
        "instruction": "Measure autoregressive decode at batch=8",
        "category": "decode",
        "difficulty": "comparison",
        "extra": {
            "phase": "decode",
            "batch_size": 8,
            "prompt_tokens": 128,
            "max_output_tokens": 128,
            "num_requests": 8,
            "description": (
                "Batched decode: amortizes weight load across requests. "
                "Expected: higher IPC than batch=1 (more compute per byte loaded). "
                "If IPC doesn't improve, BW is already saturated."
            ),
        },
    },
    {
        "id": "prefill_short",
        "instruction": "Measure prefill (prompt processing) with short prompt",
        "category": "prefill",
        "difficulty": "baseline",
        "extra": {
            "phase": "prefill",
            "batch_size": 1,
            "prompt_tokens": 256,
            "max_output_tokens": 1,
            "num_requests": 10,
            "description": (
                "Short prefill: compute-bound GEMM. "
                "Expected: high IPC (> 3.0), AMX active, Backend-Core dominant. "
                "Speculative decoding does NOT help here."
            ),
        },
    },
    {
        "id": "prefill_long",
        "instruction": "Measure prefill with long prompt (2K tokens)",
        "category": "prefill",
        "difficulty": "comparison",
        "extra": {
            "phase": "prefill",
            "batch_size": 1,
            "prompt_tokens": 2048,
            "max_output_tokens": 1,
            "num_requests": 5,
            "description": (
                "Long prefill: heavy GEMM, high arithmetic intensity. "
                "Expected: IPC > 3, low cache miss (operands fit in L3 per tile). "
                "This is the 'good' profile — no memory bottleneck."
            ),
        },
    },
    {
        "id": "mixed_realistic",
        "instruction": "Measure mixed prefill+decode with realistic distribution",
        "category": "mixed",
        "difficulty": "realistic",
        "extra": {
            "phase": "mixed",
            "batch_size": 4,
            "prompt_tokens": 512,
            "max_output_tokens": 256,
            "num_requests": 20,
            "description": (
                "Realistic serving mix: interleaved prefill and decode. "
                "Averages hide phase-specific bottlenecks — compare against "
                "pure prefill and pure decode tasks to decompose."
            ),
        },
    },
    {
        "id": "decode_speculative",
        "instruction": "Measure decode with speculative decoding enabled (draft model)",
        "category": "decode_spec",
        "difficulty": "optimization",
        "extra": {
            "phase": "decode",
            "batch_size": 1,
            "prompt_tokens": 128,
            "max_output_tokens": 128,
            "num_requests": 10,
            "speculative_model": "auto",
            "num_speculative_tokens": 5,
            "spec_method": "draft_model",
            "description": (
                "Standard speculative decode: separate draft model generates "
                "candidates, target model verifies in one pass. "
                "Expected: IPC improves over decode_batch1 (more compute per "
                "weight load), throughput 2-4× if acceptance rate > 60%."
            ),
        },
    },
    {
        "id": "decode_dflash_sd",
        "instruction": "Measure decode with DFlash speculative decoding (CPU, PR#44029)",
        "category": "decode_spec",
        "difficulty": "optimization",
        "extra": {
            "phase": "decode",
            "batch_size": 1,
            "prompt_tokens": 128,
            "max_output_tokens": 128,
            "num_requests": 10,
            "num_speculative_tokens": 8,
            "spec_method": "dflash",
            "description": (
                "DFlash speculative decoding on CPU (vllm-project/vllm#44029). "
                "Self-speculative: no separate draft model needed. Uses DFlash "
                "context KV cache with CPU-native C++ speculation logic. "
                "Reference: 19.23 tok/s on Llama-3.1-8B, acceptance rate 25%, "
                "avg acceptance length 3.01 tokens."
            ),
            "vllm_args": [
                "--speculative-method=dflash",
                "--num-speculative-tokens=8",
            ],
            "reference_pr": "https://github.com/vllm-project/vllm/pull/44029",
        },
    },
]


class VLLMInferenceAdapter:
    """BenchmarkAdapter for vLLM CPU inference with phase-aware measurement.

    Parameters
    ----------
    model : str, default="meta-llama/Llama-3.2-1B-Instruct"
        HuggingFace model ID or local path.
    mode : str, default="docker"
        How to run vLLM: "docker", "process", or "remote".
    server_url : str, optional
        URL of already-running vLLM server (mode="remote").
    docker_image : str, optional
        Docker image for vLLM CPU. Default: auto-detect.
    dtype : str, default="bfloat16"
        Model dtype: "bfloat16", "float16", "float32".
    kv_cache_gb : int, default=4
        VLLM_CPU_KVCACHE_SPACE in GB.
    tensor_parallel : int, default=1
        Tensor parallel degree (1 = single NUMA node).
    omp_threads_bind : str, default="auto"
        VLLM_CPU_OMP_THREADS_BIND setting.
    speculative_model : str, optional
        Draft model for speculative decoding tasks.
    numa_node : int, optional
        NUMA node to pin vLLM server to.
    """

    name = "vllm_inference"
    version = "1.0.0"

    def __init__(
        self,
        *,
        model: str = "meta-llama/Llama-3.2-1B-Instruct",
        mode: str = "docker",
        server_url: Optional[str] = None,
        docker_image: Optional[str] = None,
        dtype: str = "bfloat16",
        kv_cache_gb: int = 4,
        tensor_parallel: int = 1,
        omp_threads_bind: str = "auto",
        speculative_model: Optional[str] = None,
        numa_node: Optional[int] = None,
    ) -> None:
        self._model = model
        self._mode = mode
        self._server_url = server_url or "http://localhost:8000"
        self._docker_image = docker_image or "vllm/vllm-openai-cpu:latest-x86_64"
        self._dtype = dtype
        self._kv_cache_gb = kv_cache_gb
        self._tensor_parallel = tensor_parallel
        self._omp_threads_bind = omp_threads_bind
        self._speculative_model = speculative_model
        self._numa_node = numa_node
        self._server_proc: Optional[subprocess.Popen] = None
        self._container_id: Optional[str] = None

    # ─── BenchmarkAdapter Protocol ───────────────────────────────────

    def list_tasks(
        self,
        *,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> Iterable[TaskSpec]:
        """Enumerate vLLM inference benchmark tasks.

        Parameters
        ----------
        include : list of str, optional
            Filter by category: "decode", "prefill", "mixed", "decode_spec".
        exclude : list of str, optional
            Categories to exclude.
        limit : int, optional
            Maximum number of tasks.
        """
        count = 0
        for task_data in INFERENCE_TASKS:
            category = task_data["category"]

            if include and category not in include:
                continue
            if exclude and category in exclude:
                continue

            # Skip speculative tasks if no draft model configured
            if category == "decode_spec" and not self._speculative_model:
                continue

            yield TaskSpec(
                id=task_data["id"],
                instruction=task_data["instruction"],
                category=category,
                difficulty=task_data["difficulty"],
                timeout_s=300.0,
                extra=task_data["extra"],
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
        """Run one vLLM inference benchmark task.

        Starts the vLLM server if not already running, sends requests
        matching the task's phase/batch/length profile, and collects
        workload-level KPIs (tokens/s, TTFT, ITL).

        Hardware measurements (IPC, cache miss, TMA) are collected by
        AgentSysPerf's measurement plugins wrapping this call via track_span.
        """
        logger.info(
            f"Running vLLM task {task.id} "
            f"(phase={task.extra['phase']}, batch={task.extra['batch_size']})"
        )

        try:
            # Ensure server is running
            self._ensure_server_running(task)

            # Wait for server ready
            if not self._wait_for_server(timeout=120):
                return TaskResult(
                    task_id=task.id,
                    passed=False,
                    reward=0.0,
                    error="vLLM server failed to start within 120s",
                )

            # Warmup
            self._warmup(task)

            # Run benchmark requests
            results = self._run_benchmark_requests(task)

            # Compute KPIs
            kpis = self._compute_kpis(results, task)

            logger.info(
                f"Task {task.id}: "
                f"throughput={kpis['generation_tokens_per_sec']:.1f} tok/s, "
                f"TTFT_p50={kpis['ttft_p50_ms']:.1f}ms, "
                f"ITL_p50={kpis['itl_p50_ms']:.1f}ms"
            )

            return TaskResult(
                task_id=task.id,
                passed=True,
                reward=kpis["generation_tokens_per_sec"],
                extra={
                    "phase": task.extra["phase"],
                    "batch_size": task.extra["batch_size"],
                    "kpis": kpis,
                    "model": self._model,
                    "dtype": self._dtype,
                    "speculative_decoding": task.category == "decode_spec",
                    "num_requests": len(results),
                },
            )

        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}")
            return TaskResult(
                task_id=task.id,
                passed=False,
                reward=0.0,
                error=str(e),
            )

    def teardown(self) -> None:
        """Stop vLLM server and cleanup."""
        self._stop_server()
        logger.info("vLLM inference adapter teardown complete")

    # ─── Server Management ───────────────────────────────────────────

    def _ensure_server_running(self, task: TaskSpec) -> None:
        """Start vLLM server if not already running."""
        if self._mode == "remote":
            return  # Already running externally

        if self._server_proc is not None or self._container_id is not None:
            return  # Already started

        if self._mode == "docker":
            self._start_docker_server(task)
        elif self._mode == "process":
            self._start_process_server(task)

    def _start_docker_server(self, task: TaskSpec) -> None:
        """Start vLLM in Docker container.

        Flags and environment per vLLM's CPU deployment guidance
        (https://docs.vllm.ai/en/latest/getting_started/installation/cpu/):
        - --security-opt seccomp=unconfined (NUMA syscalls)
        - --cap-add SYS_NICE (thread priority)
        - VLLM_CPU_KVCACHE_SPACE configured
        - VLLM_CPU_OMP_THREADS_BIND configured
        """
        env_vars = [
            "-e", f"VLLM_CPU_KVCACHE_SPACE={self._kv_cache_gb}",
            "-e", f"VLLM_CPU_OMP_THREADS_BIND={self._omp_threads_bind}",
        ]

        cmd = [
            "docker", "run", "-d",
            "--security-opt", "seccomp=unconfined",
            "--cap-add", "SYS_NICE",
            f"--shm-size={self._kv_cache_gb + 2}g",
            "-p", "8000:8000",
            *env_vars,
        ]

        # NUMA pinning
        if self._numa_node is not None:
            cmd.extend(["--cpuset-mems", str(self._numa_node)])

        cmd.extend([
            self._docker_image,
            self._model,
            f"--dtype={self._dtype}",
            f"--tensor-parallel-size={self._tensor_parallel}",
        ])

        # Add speculative decoding args if applicable
        if task.category == "decode_spec":
            spec_method = task.extra.get("spec_method", "draft_model")
            if spec_method == "dflash":
                # DFlash SD (PR#44029): self-speculative, no draft model
                cmd.extend(task.extra.get("vllm_args", [
                    "--speculative-method=dflash",
                    f"--num-speculative-tokens={task.extra.get('num_speculative_tokens', 8)}",
                ]))
            elif self._speculative_model:
                # Standard draft-model speculative decoding
                cmd.extend([
                    f"--speculative-model={self._speculative_model}",
                    f"--num-speculative-tokens={task.extra.get('num_speculative_tokens', 5)}",
                ])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            self._container_id = result.stdout.strip()[:12]
            logger.info(f"Started vLLM Docker container: {self._container_id}")
        except Exception as e:
            raise RuntimeError(f"Failed to start vLLM Docker: {e}")

    def _start_process_server(self, task: TaskSpec) -> None:
        """Start vLLM as a local process."""
        env = os.environ.copy()
        env["VLLM_CPU_KVCACHE_SPACE"] = str(self._kv_cache_gb)
        env["VLLM_CPU_OMP_THREADS_BIND"] = self._omp_threads_bind

        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self._model,
            "--dtype", self._dtype,
            "--tensor-parallel-size", str(self._tensor_parallel),
        ]

        if task.category == "decode_spec":
            spec_method = task.extra.get("spec_method", "draft_model")
            if spec_method == "dflash":
                for arg in task.extra.get("vllm_args", ["--speculative-method=dflash"]):
                    cmd.extend(arg.split("=", 1) if "=" in arg else [arg])
            elif self._speculative_model:
                cmd.extend([
                    "--speculative-model", self._speculative_model,
                    "--num-speculative-tokens", str(task.extra.get("num_speculative_tokens", 5)),
                ])

        try:
            self._server_proc = subprocess.Popen(
                cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            logger.info(f"Started vLLM process: PID {self._server_proc.pid}")
        except Exception as e:
            raise RuntimeError(f"Failed to start vLLM process: {e}")

    def _stop_server(self) -> None:
        """Stop vLLM server."""
        if self._container_id:
            subprocess.run(
                ["docker", "stop", self._container_id],
                capture_output=True, timeout=30,
            )
            subprocess.run(
                ["docker", "rm", "-f", self._container_id],
                capture_output=True, timeout=10,
            )
            self._container_id = None

        if self._server_proc:
            self._server_proc.terminate()
            try:
                self._server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._server_proc.kill()
            self._server_proc = None

    def _wait_for_server(self, timeout: float = 120) -> bool:
        """Wait for vLLM server to be ready."""
        import urllib.request
        import urllib.error

        from src.safe_url import require_http_url

        health_url = require_http_url(f"{self._server_url}/health")
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                # scheme gated above
                req = urllib.request.urlopen(health_url, timeout=2)  # nosec B310
                if req.status == 200:
                    logger.info("vLLM server is ready")
                    return True
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(2)

        logger.error("vLLM server did not become ready")
        return False

    # ─── Benchmark Execution ─────────────────────────────────────────

    def _warmup(self, task: TaskSpec) -> None:
        """Warmup requests to fill KV cache and trigger JIT/graph capture.

        Issue enough requests to fill the KV cache and trigger JIT / graph
        capture before measuring, so first-request compilation cost does not
        land in the measured span.
        """
        logger.info("Warming up vLLM server...")
        warmup_prompt = "Hello, this is a warmup request. " * 10

        for i in range(3):
            try:
                self._send_completion_request(
                    prompt=warmup_prompt,
                    max_tokens=16,
                )
            except Exception:
                time.sleep(2)

    def _run_benchmark_requests(self, task: TaskSpec) -> List[Dict[str, Any]]:
        """Send benchmark requests and collect timing results."""
        num_requests = task.extra.get("num_requests", 10)
        prompt_tokens = task.extra.get("prompt_tokens", 128)
        max_output_tokens = task.extra.get("max_output_tokens", 128)

        # Generate prompt of approximately the right length
        # (~4 chars per token as rough estimate)
        prompt = ("The quick brown fox jumps over the lazy dog. " * 10)[:prompt_tokens * 4]

        results = []
        for i in range(num_requests):
            t_start = time.time()

            try:
                response = self._send_completion_request(
                    prompt=prompt,
                    max_tokens=max_output_tokens,
                )

                t_end = time.time()

                results.append({
                    "request_id": i,
                    "start_time": t_start,
                    "end_time": t_end,
                    "total_time_s": t_end - t_start,
                    "prompt_tokens": response.get("usage", {}).get("prompt_tokens", prompt_tokens),
                    "completion_tokens": response.get("usage", {}).get("completion_tokens", 0),
                    "ttft_s": response.get("ttft_s", 0),
                    "success": True,
                })

            except Exception as e:
                t_end = time.time()
                results.append({
                    "request_id": i,
                    "start_time": t_start,
                    "end_time": t_end,
                    "total_time_s": t_end - t_start,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "success": False,
                    "error": str(e),
                })

        return results

    def _send_completion_request(
        self, prompt: str, max_tokens: int
    ) -> Dict[str, Any]:
        """Send a completion request to vLLM OpenAI-compatible API."""
        import urllib.request

        from src.safe_url import require_http_url

        payload = json.dumps({
            "model": self._model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }).encode("utf-8")

        req = urllib.request.Request(
            require_http_url(f"{self._server_url}/v1/completions"),
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        t_first = time.time()
        # scheme gated above
        resp = urllib.request.urlopen(req, timeout=120)  # nosec B310
        response_data = json.loads(resp.read().decode("utf-8"))
        response_data["ttft_s"] = time.time() - t_first

        return response_data

    # ─── KPI Computation ─────────────────────────────────────────────

    def _compute_kpis(self, results: List[Dict[str, Any]], task: TaskSpec) -> Dict[str, Any]:
        """Compute workload KPIs from benchmark results.

        The same KPI set vLLM's own serving benchmark reports, so numbers from
        here are comparable with upstream ``benchmark_serving.py`` output:
        - Prompt tokens/s (prefill throughput)
        - Generation tokens/s (decode throughput)
        - TTFT p50/p99
        - ITL p50/p99
        """
        successful = [r for r in results if r.get("success", False)]

        if not successful:
            return {
                "generation_tokens_per_sec": 0.0,
                "prompt_tokens_per_sec": 0.0,
                "ttft_p50_ms": 0.0,
                "ttft_p99_ms": 0.0,
                "itl_p50_ms": 0.0,
                "itl_p99_ms": 0.0,
                "total_requests": len(results),
                "successful_requests": 0,
            }

        # Total tokens generated
        total_completion_tokens = sum(r["completion_tokens"] for r in successful)
        total_prompt_tokens = sum(r["prompt_tokens"] for r in successful)
        total_time = sum(r["total_time_s"] for r in successful)

        # Throughput
        gen_tok_per_sec = total_completion_tokens / total_time if total_time > 0 else 0
        prompt_tok_per_sec = total_prompt_tokens / total_time if total_time > 0 else 0

        # TTFT (time to first token)
        ttft_values = sorted([r.get("ttft_s", r["total_time_s"]) for r in successful])
        ttft_p50 = ttft_values[len(ttft_values) // 2] * 1000
        ttft_p99 = ttft_values[int(len(ttft_values) * 0.99)] * 1000

        # ITL (inter-token latency) — approximate from total time / output tokens
        itl_values = sorted([
            (r["total_time_s"] / r["completion_tokens"]) * 1000
            for r in successful if r["completion_tokens"] > 0
        ])
        itl_p50 = itl_values[len(itl_values) // 2] if itl_values else 0
        itl_p99 = itl_values[int(len(itl_values) * 0.99)] if itl_values else 0

        return {
            "generation_tokens_per_sec": gen_tok_per_sec,
            "prompt_tokens_per_sec": prompt_tok_per_sec,
            "ttft_p50_ms": ttft_p50,
            "ttft_p99_ms": ttft_p99,
            "itl_p50_ms": itl_p50,
            "itl_p99_ms": itl_p99,
            "total_requests": len(results),
            "successful_requests": len(successful),
            "total_completion_tokens": total_completion_tokens,
            "total_prompt_tokens": total_prompt_tokens,
            "total_time_s": total_time,
        }


__all__ = ["VLLMInferenceAdapter"]
