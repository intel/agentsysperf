#!/usr/bin/env python3
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Option A: One shared vLLM server + N concurrent tau-bench agents.

Architecture:
  - vLLM serves on cores 0-63 (node0 + node1), TP=2
  - N tau-bench agents run concurrently, each pinned to cores from node2 (64-91)
  - EMON collects system-wide during the workload
  - Orchestrator + EMON on reserved cores 92-95

This script:
  1. Generates and writes the vLLM launch script
  2. Optionally starts vLLM and waits for health
  3. Runs N concurrent tau-bench simulations via the TauBenchAdapter
  4. Collects per-agent latency data + EMON metrics
  5. Produces a scaling summary

Usage:
    # Generate launch script only:
    python -m experiments.scaling.launch_vllm_taubench --generate-script

    # Full run (assumes vLLM already serving on port 8000):
    python -m experiments.scaling.launch_vllm_taubench --density 4 --domain retail

    # Start vLLM + run:
    python -m experiments.scaling.launch_vllm_taubench --start-vllm --density 8

    # Sweep densities:
    python -m experiments.scaling.launch_vllm_taubench --sweep 1,2,4,8,16
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.scaling.config import (
    NUMA_NODES, ORCHESTRATOR_CORES, AGENT_POOL_CORES,
)
from experiments.scaling.vllm_cpu_config import VLLMCpuConfig, shared_vllm_config
from src.protocols import MeasurementRecord
import tempfile

# Scratch root for run artifacts. gettempdir() honours TMPDIR and falls
# back to "/tmp", so this resolves to the same paths as the hardcoded
# /tmp literals it replaced while no longer pinning output to a
# world-writable directory on systems that configure TMPDIR elsewhere.
_TMP = tempfile.gettempdir()

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

# Core partitioning for Option A
VLLM_CORES = set(range(0, 64))       # node0 (0-31) + node1 (32-63)
AGENT_CORES = set(range(64, 92))     # node2 minus reserved (28 cores for agents)

# SEP / pyEDP locations. A SEP install directory carries its version and a build
# hash in its name, so any literal path is valid on exactly one host: override via
# $SEP_DIR. /opt/intel/sep is the version-independent symlink a normal install
# leaves behind. EMON_DB must match the host microarchitecture -- TMA formulas
# differ per platform, so the wrong XML yields plausible but wrong metrics
# (`emon -v` prints the right name on its "EMON Database" line).
SEP_DIR = Path(os.environ.get("SEP_DIR", "/opt/intel/sep"))
EMON_BIN = SEP_DIR / "bin64" / "emon"
PYEDP_DIR = SEP_DIR / "config" / "edp"
EMON_DB = os.environ.get("EMON_DB", "graniterapids_server")
METRICS_XML = PYEDP_DIR / f"{EMON_DB}.xml"
CHART_FORMAT = PYEDP_DIR / f"chart_format_{EMON_DB}.txt"


def _wait_for_vllm(base_url: str, timeout: int = 300) -> bool:
    """Poll vLLM health endpoint until ready."""
    import urllib.request
    import urllib.error

    from src.safe_url import require_http_url

    health_url = require_http_url(f"{base_url}/health")
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(health_url, method="GET")
            # scheme gated above
            with urllib.request.urlopen(req, timeout=5) as resp:  # nosec B310
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    return False


def _start_emon(output_file: Path) -> Optional[subprocess.Popen]:
    """Start EMON EDP collection."""
    if not EMON_BIN.exists():
        return None
    cmd = [str(EMON_BIN), "-collect-edp", "-f", str(output_file)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(0.5)
    return proc


def _stop_emon(proc: Optional[subprocess.Popen]) -> None:
    """Stop EMON collection."""
    if proc is None:
        return
    subprocess.run([str(EMON_BIN), "-stop"], capture_output=True, timeout=10)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _post_process_emon(dat_file: Path) -> Optional[Path]:
    """Run pyEDP post-processing on EMON .dat file. Returns CSV path or None."""
    if not dat_file.exists():
        return None

    csv_prefix = dat_file.with_suffix(".csv")
    env = os.environ.copy()
    env["PATH"] = f"{SEP_DIR / 'bin64'}:{env.get('PATH', '')}"
    env["PYTHONPATH"] = f"{PYEDP_DIR}:{PYEDP_DIR / 'pyedp'}:{env.get('PYTHONPATH', '')}"

    cmd = [
        sys.executable, "-m", "pyedp.mpp",
        "--socket-view",
        "-i", str(dat_file),
        "-o", str(csv_prefix),
        "-m", str(METRICS_XML),
        "-f", str(CHART_FORMAT),
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    for suffix in ("_system_view_summary.csv", "_system_view_details.csv",
                   "_socket_view_summary.csv", "_socket_view_details.csv"):
        candidate = Path(f"{csv_prefix}{suffix}")
        if candidate.exists():
            return candidate
    return None


def _analyze_emon(emon_dat: Path, run_dir: Path) -> Optional[Dict[str, Any]]:
    """Run the full EMON 5-step pipeline: pyEDP + EmonAnalyzer.

    Returns analysis dict with TMA breakdown, findings, and root causes,
    or None if EMON data is unavailable or processing fails.
    """
    if not emon_dat.exists():
        return None

    print(f"  EMON: post-processing ({emon_dat.stat().st_size // 1024} KB)...")
    csv_path = _post_process_emon(emon_dat)
    if csv_path is None:
        print(f"  EMON: pyEDP produced no CSV output")
        return None

    print(f"  EMON: metrics CSV → {csv_path.name}")

    try:
        from agentsysperf_emon.analyzer import (
            EmonAnalyzer, load_emon_csv_auto, triage_workload,
        )

        # Step 1: Load
        data = load_emon_csv_auto(str(csv_path))

        # Step 2: Triage
        workload_class, signals = triage_workload(data)
        print(f"  EMON: workload class = {workload_class.upper()}")

        # Steps 3-5: Layer analysis → findings → report
        analyzer = EmonAnalyzer()
        record = MeasurementRecord(
            span_id="tau_bench_scaling_run",
            layer="emon",
            payload={"csv_path": str(csv_path)},
        )

        results = list(analyzer.analyze([record]))
        if not results:
            return {"workload_class": workload_class, "signals": signals,
                    "csv_path": str(csv_path)}

        evidence = results[0].evidence
        return {
            "workload_class": workload_class,
            "signals": signals,
            "csv_path": str(csv_path),
            "tma_top_level": evidence.get("tma_top_level", {}),
            "total_findings": evidence.get("total_findings", 0),
            "actionable_findings": evidence.get("actionable_findings", 0),
            "addressable_gain": evidence.get("addressable_gain", ""),
            "root_causes": evidence.get("root_causes", []),
            "recommendations": list(results[0].recommendations),
        }

    except Exception as e:
        print(f"  EMON: analysis failed — {e}")
        return None


def _bridge_latencies_to_phase_records(agent_results: List[Dict]) -> List[MeasurementRecord]:
    """Convert LatencyRecorder per-agent data into PhaseProfiler-compatible MeasurementRecords.

    Each agent result may have 'extra' containing llm_calls[] and tool_calls[]
    from the TauBenchAdapter's LatencyRecorder. These map to:
      - llm_calls → phase "reason"
      - tool_calls → phase "act"
    """
    records: List[MeasurementRecord] = []
    span_idx = 0

    for agent in agent_results:
        if "error" in agent and agent.get("error"):
            continue
        agent_id = agent.get("agent_id", 0)

        for task_result in agent.get("results", []):
            task_extra = task_result.get("extra", {})
            if not task_extra:
                continue

            # Try individual call records first (from LatencyRecorder dump)
            llm_calls = task_extra.get("llm_calls", [])
            tool_calls = task_extra.get("tool_calls", [])

            if llm_calls:
                for call in llm_calls:
                    latency_s = call.get("latency_s", 0) if isinstance(call, dict) else 0
                    span_id = f"agent_{agent_id}/turn_{span_idx}_llm"
                    records.append(MeasurementRecord(
                        span_id=span_id,
                        layer="l1",
                        payload={
                            "phase": "reason",
                            "wall_s": latency_s,
                            "cpu_s": latency_s * 0.95,
                            "duration_us": int(latency_s * 1_000_000),
                            "cpu_time_s": latency_s * 0.95,
                        },
                    ))
                    span_idx += 1
            elif task_extra.get("num_llm_calls"):
                # Synthesize from summary stats
                n = task_extra["num_llm_calls"]
                stats = task_extra.get("llm_latency_stats", {})
                mean_s = stats.get("mean", 1.0) if stats else 1.0
                for i in range(n):
                    span_id = f"agent_{agent_id}/turn_{span_idx}_llm"
                    records.append(MeasurementRecord(
                        span_id=span_id,
                        layer="l1",
                        payload={
                            "phase": "reason",
                            "wall_s": mean_s,
                            "cpu_s": mean_s * 0.95,
                            "duration_us": int(mean_s * 1_000_000),
                            "cpu_time_s": mean_s * 0.95,
                        },
                    ))
                    span_idx += 1

            if tool_calls:
                for call in tool_calls:
                    latency_s = call.get("latency_s", 0) if isinstance(call, dict) else 0
                    span_id = f"agent_{agent_id}/turn_{span_idx}_cmd"
                    records.append(MeasurementRecord(
                        span_id=span_id,
                        layer="l1",
                        payload={
                            "phase": "act",
                            "wall_s": latency_s,
                            "cpu_s": latency_s * 0.5,
                            "duration_us": int(latency_s * 1_000_000),
                            "cpu_time_s": latency_s * 0.5,
                        },
                    ))
                    span_idx += 1
            elif task_extra.get("num_tool_calls"):
                n = task_extra["num_tool_calls"]
                stats = task_extra.get("tool_latency_stats", {})
                mean_s = stats.get("mean", 0.2) if stats else 0.2
                for i in range(n):
                    span_id = f"agent_{agent_id}/turn_{span_idx}_cmd"
                    records.append(MeasurementRecord(
                        span_id=span_id,
                        layer="l1",
                        payload={
                            "phase": "act",
                            "wall_s": mean_s,
                            "cpu_s": mean_s * 0.5,
                            "duration_us": int(mean_s * 1_000_000),
                            "cpu_time_s": mean_s * 0.5,
                        },
                    ))
                    span_idx += 1

    return records


def _run_tau_agent(
    agent_id: int,
    domain: str,
    model_name: str,
    vllm_base_url: str,
    num_tasks: int,
    output_dir: Path,
    cpuset: set[int],
) -> dict:
    """Run one tau-bench agent (called in a thread)."""
    # Pin this thread's future work to a cpuset
    try:
        os.sched_setaffinity(0, cpuset)
    except (OSError, AttributeError):
        pass

    from src.benchmarks.tau_bench.adapter import TauBenchAdapter

    agent_output = output_dir / f"agent_{agent_id}"
    adapter = TauBenchAdapter(
        domain=domain,
        model_name=model_name,
        vllm_base_url=vllm_base_url,
        num_trials=1,
        max_steps=100,
        max_concurrency=1,
        output_dir=agent_output,
        temperature=0.0,
    )

    tasks = list(adapter.list_tasks(limit=num_tasks))
    results = []
    t0 = time.time()

    for task in tasks:
        # Dummy invoker (tau2 manages its own LLM calls)
        class _NoOpInvoker:
            def invoke(self, instruction, **kwargs):
                return ""

        result = adapter.run_task(task, agent_invoker=_NoOpInvoker())
        results.append({
            "task_id": result.task_id,
            "passed": result.passed,
            "reward": result.reward,
            "error": result.error,
            "extra": dict(result.extra) if result.extra else {},
        })

    wall_time = time.time() - t0
    adapter.teardown()

    return {
        "agent_id": agent_id,
        "domain": domain,
        "cpuset": sorted(cpuset),
        "num_tasks": len(tasks),
        "wall_time_s": wall_time,
        "tasks_per_s": len(tasks) / wall_time if wall_time > 0 else 0,
        "results": results,
    }


@app.command()
def main(
    density: int = typer.Option(1, "--density", "-n", help="Number of concurrent tau-bench agents"),
    domain: str = typer.Option("retail", "--domain", "-d", help="tau2 domain"),
    model: str = typer.Option(
        "hosted_vllm/Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "--model", "-m", help="LiteLLM model name",
    ),
    vllm_base_url: str = typer.Option("http://localhost:8000/v1", "--vllm-url"),
    num_tasks: int = typer.Option(3, "--num-tasks", help="Tasks per agent"),
    output: Path = typer.Option(Path(f"{_TMP}/agentsysperf_taubench_scaling"), "--output", "-o"),
    start_vllm: bool = typer.Option(False, "--start-vllm", help="Launch vLLM before running"),
    vllm_model: str = typer.Option(
        "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "--vllm-model", help="Model for vLLM serve (if --start-vllm)",
    ),
    generate_script: bool = typer.Option(False, "--generate-script", help="Only generate launch script"),
    sweep: Optional[str] = typer.Option(None, "--sweep", help="Comma-separated densities to sweep"),
    collect_emon: bool = typer.Option(True, "--emon/--no-emon", help="Collect EMON data"),
) -> None:
    """Run N concurrent tau-bench agents against a shared vLLM server."""

    # Generate launch script
    cfg = shared_vllm_config(model=vllm_model, vllm_cores=VLLM_CORES)
    script_content = cfg.launch_script(vllm_cores=VLLM_CORES)
    output.mkdir(parents=True, exist_ok=True)
    script_path = output / "launch_vllm.sh"
    script_path.write_text(script_content)
    script_path.chmod(0o755)

    if generate_script:
        print(f"Generated: {script_path}")
        print(f"\nTo start vLLM:")
        print(f"  bash {script_path}")
        return

    # Start vLLM if requested
    vllm_proc = None
    if start_vllm:
        print(f"Starting vLLM (model: {vllm_model})...")
        env = os.environ.copy()
        env.update(cfg.env_vars(VLLM_CORES))
        vllm_proc = subprocess.Popen(
            cfg.serve_command(VLLM_CORES),
            env=env,
            stdout=open(output / "vllm_stdout.log", "w"),
            stderr=open(output / "vllm_stderr.log", "w"),
        )
        print(f"  PID: {vllm_proc.pid}, waiting for health...")
        base_url_no_v1 = vllm_base_url.replace("/v1", "")
        if not _wait_for_vllm(base_url_no_v1, timeout=300):
            print("  ERROR: vLLM failed to start within 5 minutes")
            vllm_proc.terminate()
            raise typer.Exit(1)
        print("  vLLM ready!")

    # Determine densities to run
    densities = [density]
    if sweep:
        densities = [int(x) for x in sweep.split(",")]

    try:
        for d in densities:
            _run_density(
                density=d,
                domain=domain,
                model_name=model,
                vllm_base_url=vllm_base_url,
                num_tasks=num_tasks,
                output_dir=output,
                collect_emon=collect_emon,
            )
    finally:
        if vllm_proc:
            print("\nStopping vLLM...")
            vllm_proc.send_signal(signal.SIGTERM)
            try:
                vllm_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                vllm_proc.kill()


def _run_density(
    density: int,
    domain: str,
    model_name: str,
    vllm_base_url: str,
    num_tasks: int,
    output_dir: Path,
    collect_emon: bool,
) -> None:
    """Run one density level: N concurrent agents."""
    run_dir = output_dir / f"density_{density}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Density: {density} concurrent tau-bench agents")
    print(f"  Domain: {domain}, Tasks/agent: {num_tasks}")
    print(f"  vLLM: {vllm_base_url}")
    print(f"{'='*60}")

    # Assign cores to agents from AGENT_CORES pool (round-robin)
    available = sorted(AGENT_CORES)
    cores_per_agent = max(1, len(available) // density) if density > 0 else len(available)
    agent_cpusets = []
    for i in range(density):
        start = (i * cores_per_agent) % len(available)
        cpuset = set(available[start:start + cores_per_agent])
        if not cpuset:
            cpuset = {available[i % len(available)]}
        agent_cpusets.append(cpuset)

    print(f"  Cores/agent: {cores_per_agent} (from pool of {len(available)})")

    # Pin orchestrator
    try:
        os.sched_setaffinity(0, ORCHESTRATOR_CORES)
    except (OSError, AttributeError):
        pass

    # Start EMON
    emon_proc = None
    emon_dat = run_dir / "emon.dat"
    if collect_emon:
        emon_proc = _start_emon(emon_dat)
        if emon_proc:
            print(f"  EMON: collecting (PID {emon_proc.pid})")
        time.sleep(2)  # warmup

    # Launch N agents concurrently
    t0 = time.time()
    agent_results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=density) as executor:
        futures = {}
        for i in range(density):
            fut = executor.submit(
                _run_tau_agent,
                agent_id=i,
                domain=domain,
                model_name=model_name,
                vllm_base_url=vllm_base_url,
                num_tasks=num_tasks,
                output_dir=run_dir,
                cpuset=agent_cpusets[i],
            )
            futures[fut] = i

        for fut in concurrent.futures.as_completed(futures):
            agent_id = futures[fut]
            try:
                result = fut.result()
                agent_results.append(result)
                print(f"  Agent {agent_id}: {result['num_tasks']} tasks in "
                      f"{result['wall_time_s']:.1f}s ({result['tasks_per_s']:.2f} tasks/s)")
            except Exception as e:
                print(f"  Agent {agent_id}: ERROR - {e}")
                agent_results.append({"agent_id": agent_id, "error": str(e)})

    total_wall = time.time() - t0

    # Stop EMON
    if emon_proc:
        time.sleep(1)  # cooldown
        _stop_emon(emon_proc)
        print(f"  EMON: stopped")

    # Summary
    successful = [r for r in agent_results if "error" not in r or r.get("error") is None]
    print(f"\n  Summary (density={density}):")
    print(f"    Wall time: {total_wall:.1f}s")
    print(f"    Successful agents: {len(successful)}/{density}")
    if successful:
        mean_tps = sum(r["tasks_per_s"] for r in successful) / len(successful)
        aggregate_tps = sum(r["tasks_per_s"] for r in successful)
        mean_wall = sum(r["wall_time_s"] for r in successful) / len(successful)
        print(f"    Mean tasks/s per agent: {mean_tps:.3f}")
        print(f"    Aggregate tasks/s: {aggregate_tps:.3f}")
        print(f"    Mean wall time: {mean_wall:.1f}s")

    # ─── EMON 5-step analysis pipeline ───────────────────────────────────
    emon_analysis = None
    if collect_emon and emon_dat.exists():
        emon_analysis = _analyze_emon(emon_dat, run_dir)

    # ─── PhaseProfiler: bridge latencies → phase breakdown ───────────────
    phase_breakdown = None
    phase_recommendations: List[str] = []
    try:
        phase_records = _bridge_latencies_to_phase_records(agent_results)
        if phase_records:
            from src.analyzers.phase_profiler import PhaseProfiler, PHASE_LABELS
            profiler = PhaseProfiler()
            phase_results = list(profiler.analyze(phase_records))
            if phase_results:
                phase_breakdown = phase_results[0].evidence.get("phase_breakdown")
                phase_recommendations = list(phase_results[0].recommendations)
    except Exception as e:
        print(f"  PhaseProfiler: analysis failed — {e}")

    # ─── Unified report ──────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  UNIFIED ANALYSIS (density={density})")
    print(f"{'─'*60}")

    if emon_analysis:
        tma = emon_analysis.get("tma_top_level", {})
        if any(v > 0 for v in tma.values()):
            print(f"\n    EMON TMA Breakdown:")
            for k, v in sorted(tma.items(), key=lambda x: -x[1]):
                if v > 0:
                    print(f"      {k:22s} {v:5.1f}%")
        root_causes = emon_analysis.get("root_causes", [])
        if root_causes:
            print(f"\n    Root Causes (top {min(3, len(root_causes))}):")
            for rc in root_causes[:3]:
                print(f"      [{rc['severity']:8s}] {rc['headline']}")
                print(f"        Fix: {rc['fix']} | Gain: {rc['gain_range']}")

    if phase_breakdown:
        print(f"\n    PhaseProfiler Breakdown:")
        print(f"      {'Phase':<10} {'Wall%':>7} {'CPU%':>7} {'WallMs':>8}")
        for phase, pdata in phase_breakdown.items():
            print(f"      {phase:<10} {pdata['wall_pct']:>6.1f}% "
                  f"{pdata['cpu_pct']:>6.1f}% {pdata['wall_ms']:>7.1f}")

    if successful:
        print(f"\n    Per-Agent Latency:")
        print(f"      Mean wall time:   {mean_wall:.1f}s")
        print(f"      Mean tasks/s:     {mean_tps:.3f}")
        print(f"      Aggregate tasks/s:{aggregate_tps:.3f}")

    if phase_recommendations:
        print(f"\n    Recommendations:")
        for rec in phase_recommendations[:3]:
            print(f"      - {rec}")

    print(f"{'─'*60}")

    # Save results
    summary: Dict[str, Any] = {
        "density": density,
        "domain": domain,
        "model": model_name,
        "num_tasks_per_agent": num_tasks,
        "total_wall_time_s": total_wall,
        "cores_per_agent": cores_per_agent,
        "vllm_cores": sorted(VLLM_CORES),
        "agent_cores": sorted(AGENT_CORES),
        "agent_results": agent_results,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"    Results: {run_dir / 'summary.json'}")

    # Save unified analysis
    analysis: Dict[str, Any] = {
        "density": density,
        "domain": domain,
        "total_wall_time_s": total_wall,
        "emon": emon_analysis,
        "phase_breakdown": phase_breakdown,
        "phase_recommendations": phase_recommendations,
        "per_agent_summary": {
            "successful": len(successful),
            "total": density,
            "mean_tasks_per_s": mean_tps if successful else 0,
            "aggregate_tasks_per_s": aggregate_tps if successful else 0,
            "mean_wall_time_s": mean_wall if successful else 0,
        },
    }
    (run_dir / "analysis.json").write_text(json.dumps(analysis, indent=2, default=str))
    print(f"    Analysis: {run_dir / 'analysis.json'}")


if __name__ == "__main__":
    app()
