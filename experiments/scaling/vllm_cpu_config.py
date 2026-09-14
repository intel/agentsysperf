#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""vLLM CPU configuration for Granite Rapids Xeon (96C/192T, SNC3).

Encodes the optimal vLLM settings from:
  - https://docs.vllm.ai/en/latest/getting_started/installation/cpu/
  - https://docs.vllm.ai/en/latest/benchmarking/dashboard/

Key decisions:
  - tensor-parallel-size=3 (one TP rank per NUMA node under SNC3)
  - bfloat16 (float16 is unstable on CPU)
  - VLLM_CPU_SGL_KERNEL=1 (small-batch optimized kernel, helps low-concurrency)
  - VLLM_CPU_KVCACHE_SPACE sized for long agentic conversations (100+ steps)
  - OMP thread binding aligned with NUMA topology and experiment core reservations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .config import NUMA_NODES, ORCHESTRATOR_CORES, AGENT_POOL_CORES


@dataclass
class VLLMCpuConfig:
    """Configuration for launching vLLM on CPU for the scaling experiment."""

    model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    port: int = 8000
    # Loopback, not 0.0.0.0. vLLM's OpenAI server has no auth, so binding all
    # interfaces hands anyone routable to this box free inference on the SKU
    # we are trying to measure — which also corrupts the measurement. The only
    # client in this experiment is local (launch_vllm_taubench defaults
    # --vllm-url to http://localhost:8000/v1). Set host="0.0.0.0" explicitly if
    # you are driving the server from another node.
    host: str = "127.0.0.1"
    dtype: str = "bfloat16"

    # Tensor parallelism: 3 for SNC3 (one rank per NUMA node)
    tensor_parallel_size: int = 3

    # KV cache space in GiB — sized for long agentic conversations
    # 100-step tau-bench conversations with 30B model need ~40GB KV cache
    kvcache_space_gib: int = 40

    # Block size: multiples of 32, default 128
    block_size: int = 128

    # Batching limits (scaled by tensor_parallel_size per vLLM docs)
    # Online serving: 2048 * world_size, 128 * world_size
    max_num_batched_tokens: Optional[int] = None  # auto = 2048 * tp
    max_num_seqs: Optional[int] = None  # auto = 128 * tp

    # Max model length (for Qwen3-Coder with 131072 context)
    max_model_len: int = 131072

    # GPU memory utilization (irrelevant on CPU, but vLLM requires it)
    gpu_memory_utilization: float = 0.95

    # Enable auto tool choice for tau-bench function-calling
    enable_auto_tool_choice: bool = True
    tool_call_parser: str = "qwen3_coder"

    # CPU-specific kernel optimizations
    sgl_kernel: bool = True  # x86 small-batch optimized kernels

    # Core binding for vLLM (pipe-separated for TP ranks)
    # Default: use all agent pool cores split across NUMA nodes
    omp_threads_bind: Optional[str] = None  # None = compute from topology

    # Reserved cores NOT used by vLLM (for EMON, orchestrator)
    reserved_cores: int = 4

    def effective_omp_bind(self, vllm_cores: Optional[set[int]] = None) -> str:
        """Compute OMP thread binding string.

        For TP=3 on SNC3, pipe-separates core ranges per NUMA node:
          "0-31|32-63|64-91"

        If vllm_cores is provided (subset for coexistence with agents),
        splits those cores across TP ranks evenly.
        """
        if self.omp_threads_bind is not None:
            return self.omp_threads_bind

        if vllm_cores is not None:
            cores = sorted(vllm_cores)
        else:
            cores = sorted(AGENT_POOL_CORES)

        if self.tensor_parallel_size == 1:
            return f"{cores[0]}-{cores[-1]}"

        # Split cores into TP ranks by NUMA node membership
        node_cores: dict[int, list[int]] = {n: [] for n in NUMA_NODES}
        for c in cores:
            for node_id, node_core_list in NUMA_NODES.items():
                if c in node_core_list:
                    node_cores[node_id].append(c)
                    break

        # Build pipe-separated ranges
        parts = []
        for node_id in sorted(node_cores.keys()):
            nc = node_cores[node_id]
            if nc:
                parts.append(f"{nc[0]}-{nc[-1]}")

        return "|".join(parts) if parts else "auto"

    def effective_batched_tokens(self) -> int:
        if self.max_num_batched_tokens is not None:
            return self.max_num_batched_tokens
        return 2048 * self.tensor_parallel_size

    def effective_max_seqs(self) -> int:
        if self.max_num_seqs is not None:
            return self.max_num_seqs
        return 128 * self.tensor_parallel_size

    def env_vars(self, vllm_cores: Optional[set[int]] = None) -> dict[str, str]:
        """Environment variables to set before launching vLLM."""
        env = {
            "VLLM_CPU_KVCACHE_SPACE": str(self.kvcache_space_gib),
            "VLLM_CPU_OMP_THREADS_BIND": self.effective_omp_bind(vllm_cores),
            "VLLM_CPU_NUM_OF_RESERVED_CPU": str(self.reserved_cores),
            "VLLM_TARGET_DEVICE": "cpu",
        }
        if self.sgl_kernel:
            env["VLLM_CPU_SGL_KERNEL"] = "1"
        return env

    def serve_command(self, vllm_cores: Optional[set[int]] = None) -> list[str]:
        """Build the `vllm serve` command line."""
        cmd = [
            "vllm", "serve", self.model,
            "--host", self.host,
            "--port", str(self.port),
            "--dtype", self.dtype,
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--block-size", str(self.block_size),
            "--max-num-batched-tokens", str(self.effective_batched_tokens()),
            "--max-num-seqs", str(self.effective_max_seqs()),
            "--max-model-len", str(self.max_model_len),
        ]
        if self.enable_auto_tool_choice:
            cmd.extend(["--enable-auto-tool-choice", "--tool-call-parser", self.tool_call_parser])
        return cmd

    def launch_script(self, vllm_cores: Optional[set[int]] = None) -> str:
        """Generate a bash launch script."""
        env = self.env_vars(vllm_cores)
        cmd = self.serve_command(vllm_cores)

        lines = [
            "#!/bin/bash",
            "# vLLM CPU launch script for Granite Rapids Xeon (96C, SNC3)",
            "# Generated by AgentSysPerf scaling experiment",
            "",
            "set -euo pipefail",
            "",
            "# ─── CPU-specific environment ───────────────────────────────────",
        ]
        for k, v in env.items():
            lines.append(f'export {k}="{v}"')

        lines.extend([
            "",
            "# ─── Proxy bypass for local serving ─────────────────────────────",
            'export no_proxy="${no_proxy:+$no_proxy,}localhost,127.0.0.1"',
            'export NO_PROXY="${NO_PROXY:+$NO_PROXY,}localhost,127.0.0.1"',
            "",
            "# ─── Launch vLLM ────────────────────────────────────────────────",
            f"echo 'Starting vLLM on port {self.port} with TP={self.tensor_parallel_size}'",
            f"echo 'Model: {self.model}'",
            f"echo 'KV Cache: {self.kvcache_space_gib} GiB'",
            f"echo 'OMP Bind: {self.effective_omp_bind(vllm_cores)}'",
            "echo 'SGL Kernel: {}'".format("enabled" if self.sgl_kernel else "disabled"),
            "",
            " ".join(cmd),
        ])
        return "\n".join(lines) + "\n"


# ─── Presets ─────────────────────────────────────────────────────────────────

def solo_vllm_config(model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct") -> VLLMCpuConfig:
    """Config for solo benchmarking: vLLM gets all 92 agent-pool cores."""
    return VLLMCpuConfig(
        model=model,
        tensor_parallel_size=3,
        kvcache_space_gib=40,
        sgl_kernel=True,
    )


def shared_vllm_config(
    model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    vllm_cores: Optional[set[int]] = None,
) -> VLLMCpuConfig:
    """Config for Option A: one shared vLLM, N agents hit it concurrently.

    vLLM gets a dedicated core partition; agents get the rest.
    Default: vLLM on cores 0-47 (node0 + half node1), agents on 48-91.
    """
    cfg = VLLMCpuConfig(
        model=model,
        tensor_parallel_size=2,  # 2 NUMA nodes for vLLM
        kvcache_space_gib=40,
        sgl_kernel=True,
        max_num_seqs=384,  # support up to 32 concurrent agent requests
    )
    if vllm_cores is not None:
        cfg.omp_threads_bind = cfg.effective_omp_bind(vllm_cores)
    return cfg
