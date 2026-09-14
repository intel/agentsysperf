#!/usr/bin/env python3
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""
Local runner for AgentSysPerf harness — drives Harbor/Terminus-2 + TB2 on this host.

Replaces the AWS SSM-based runner from agentic-benchmark-4 with direct
subprocess execution. Orchestrates:
  1. Start replay proxy (configured mode)
  2. Start monitoring (mpstat, vmstat, perf if available, Docker stats)
  3. Run Harbor with TB2 workload
  4. Stop monitoring, collect results
  5. Post-process: join perf samples with turn timestamps for stage inference

Usage:
    python runner.py                     # uses config/local.yml defaults
    python runner.py --llm-mode off      # override LLM mode
    python runner.py --profile isolated  # run with a specific optimization profile
    python runner.py --tasks 3           # subset of tasks for quick iteration
"""
import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

HARNESS_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = HARNESS_ROOT / 'scripts'
CONFIG_DIR = HARNESS_ROOT / 'config'
RESULTS_DIR = HARNESS_ROOT / 'results'

sys.path.insert(0, str(SCRIPTS_DIR))
from perf_sampler import PerfSampler, PerfConfig, mpstat_sampler, vmstat_sampler


def load_config(path: Optional[Path] = None) -> dict:
    import yaml
    path = path or (CONFIG_DIR / 'local.yml')
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass
class RunSpec:
    label: str
    llm_mode: str
    profile_name: str
    concurrency: int
    attempts: int
    tasks: list
    agent_timeout_mult: float
    output_dir: Path


def start_proxy(cfg: dict, llm_mode: str) -> Optional[subprocess.Popen]:
    """Start the replay proxy in the configured mode."""
    proxy_cfg = cfg.get('proxy', {})
    port = proxy_cfg.get('port', 4001)

    cmd = [
        sys.executable, str(SCRIPTS_DIR / 'replay_proxy.py'),
        '--mode', llm_mode,
        '--port', str(port),
    ]

    if llm_mode == 'replay':
        fixture = proxy_cfg.get('fixture_file', '')
        fixture_path = (HARNESS_ROOT / fixture).resolve()
        if not fixture_path.exists():
            print(f'ERROR: fixture not found at {fixture_path}')
            return None
        cmd += ['--fixture', str(fixture_path)]
    elif llm_mode == 'record':
        live_cfg = cfg.get('live', {})
        upstream = live_cfg.get('endpoint', 'http://127.0.0.1:4000')
        cmd += ['--upstream', upstream]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(2)

    # Health check
    import httpx
    try:
        r = httpx.get(f'http://127.0.0.1:{port}/healthz', timeout=5)
        health = r.json()
        print(f'  Proxy started: mode={health["mode"]} '
              f'trials={health["fixture_trials"]} entries={health["fixture_entries"]}')
        return proc
    except Exception as e:
        print(f'  Proxy health check failed: {e}')
        proc.kill()
        return None


def stop_proxy(proc: Optional[subprocess.Popen]):
    if proc:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def start_docker_monitor(output_path: Path, interval_s: int = 1) -> subprocess.Popen:
    """Monitor per-container CPU% and memory, 1s cadence."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    script = f"""#!/bin/bash
while true; do
    ts=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
    docker stats --no-stream --format '{{{{.Name}}}} {{{{.CPUPerc}}}} {{{{.MemUsage}}}} {{{{.PIDs}}}}' 2>/dev/null | while read line; do
        echo "$ts $line"
    done
    sleep {interval_s}
done
"""
    proc = subprocess.Popen(
        ['bash', '-c', script],
        stdout=open(output_path, 'w'),
        stderr=subprocess.DEVNULL,
    )
    return proc


def start_proxy_monitor(output_path: Path, port: int = 4001,
                        interval_s: float = 0.5) -> subprocess.Popen:
    """Monitor proxy calls — log each completion request with timestamps.

    The proxy already logs, but this gives us a structured feed of turn
    boundaries we can join with perf samples.
    """
    # We'll capture this from the proxy's own logs instead — simpler.
    # This function is a placeholder for an explicit turn-event collector
    # if the proxy logs aren't sufficient.
    return None


def run_harbor(spec: RunSpec, proxy_port: int) -> tuple[str, float]:
    """Execute Harbor with TB2 workload. Returns (stdout_tail, elapsed_s)."""

    api_base = f'http://127.0.0.1:{proxy_port}/v1'
    include_flags = ' '.join(f'-i terminal-bench/{t}' for t in spec.tasks)

    env = os.environ.copy()
    env.update({
        'OPENAI_API_KEY': 'sk-not-needed-local-only',
        'OPENAI_API_BASE': api_base,
        'OPENAI_BASE_URL': api_base,
    })

    # Build the harbor command
    jobs_dir = spec.output_dir / 'jobs'
    cmd = (
        f'harbor run '
        f'-d terminal-bench/terminal-bench-2 '
        f'-a terminus-2 '
        f'-m "openai/agentsysperf-proxy" '
        f'-n {spec.concurrency} -k {spec.attempts} '
        f'{include_flags} '
        f'-o {jobs_dir} '
        f'--ae "OPENAI_API_BASE={api_base}" '
        f'--ae "OPENAI_API_KEY=sk-not-needed-local-only" '
        f'--ae "OPENAI_BASE_URL={api_base}" '
        f'--agent-timeout-multiplier {spec.agent_timeout_mult} '
        f'--agent-setup-timeout-multiplier 3.0 '
        f'--no-delete -y'
    )

    print(f'  Running: {cmd[:120]}...')
    t0 = time.time()
    # shlex.split, not shell=True: one `harbor run` invocation, no pipeline or
    # redirect, so the argv is identical and task names / paths interpolated
    # into `cmd` above stop being shell metacharacters.
    result = subprocess.run(
        shlex.split(cmd), env=env,
        capture_output=True, text=True,
        timeout=7200,  # 2hr max
    )
    elapsed = time.time() - t0

    # Save full output
    (spec.output_dir / 'harbor_stdout.txt').write_text(result.stdout)
    (spec.output_dir / 'harbor_stderr.txt').write_text(result.stderr)

    tail = '\n'.join(result.stdout.strip().split('\n')[-20:])
    return tail, elapsed


def collect_task_phases(jobs_dir: Path, output_path: Path) -> list:
    """Extract per-trial timing from Harbor result.json files."""
    phases = []
    if not jobs_dir.exists():
        return phases

    for job_dir in sorted(jobs_dir.iterdir()):
        if not job_dir.is_dir():
            continue
        for sub in sorted(job_dir.iterdir()):
            rf = sub / 'result.json'
            if not rf.exists():
                continue
            try:
                d = json.loads(rf.read_text())
                ae = d.get('agent_execution') or {}
                vr = d.get('verifier_result') or {}
                rewards = vr.get('rewards') or {}
                reward = rewards.get('reward', 0.0) if isinstance(rewards, dict) else 0.0
                task_name = d.get('task_name') or sub.name
                phases.append({
                    'task': task_name,
                    'started_at': ae.get('started_at', ''),
                    'finished_at': ae.get('finished_at', ''),
                    'reward': reward,
                    'duration_s': ae.get('duration_s', 0),
                    'turns': ae.get('total_turns', 0),
                    'tool_calls': ae.get('total_tool_calls', 0),
                })
            except Exception as e:
                print(f'    Warning: could not parse {rf}: {e}')

    output_path.write_text(json.dumps(phases, indent=2))
    print(f'  Collected {len(phases)} trial phases')
    return phases


def write_stats(spec: RunSpec, elapsed: float, phases: list):
    """Write summary stats.txt."""
    stats = spec.output_dir / 'stats.txt'
    total_trials = len(spec.tasks) * spec.attempts
    completed = len(phases)
    rewards = [p['reward'] for p in phases if p.get('reward')]
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0

    with open(stats, 'w') as f:
        f.write(f'label: {spec.label}\n')
        f.write(f'llm_mode: {spec.llm_mode}\n')
        f.write(f'profile: {spec.profile_name}\n')
        f.write(f'concurrency: {spec.concurrency}\n')
        f.write(f'attempts: {spec.attempts}\n')
        f.write(f'total_trials: {total_trials}\n')
        f.write(f'completed_trials: {completed}\n')
        f.write(f'elapsed_s: {elapsed:.1f}\n')
        f.write(f'avg_reward: {avg_reward:.3f}\n')
        f.write(f'tasks: {",".join(spec.tasks)}\n')


def run_one(cfg: dict, llm_mode: str, profile_name: str,
            task_subset: Optional[int] = None) -> str:
    """Execute one full benchmark cell."""

    workload = cfg.get('workload', {})
    tasks = workload.get('tasks', [])
    if task_subset:
        tasks = tasks[:task_subset]

    conc = workload.get('concurrency', 5)
    attempts = workload.get('attempts', 1)
    timeout_mult = workload.get('agent_timeout_multiplier', 2.0)
    proxy_port = cfg.get('proxy', {}).get('port', 4001)

    label = f'{profile_name}_{llm_mode}_c{conc}'
    output_dir = RESULTS_DIR / label
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = RunSpec(
        label=label, llm_mode=llm_mode, profile_name=profile_name,
        concurrency=conc, attempts=attempts, tasks=tasks,
        agent_timeout_mult=timeout_mult, output_dir=output_dir,
    )

    print(f'\n{"="*60}')
    print(f'RUN: {label}')
    print(f'  Tasks: {len(tasks)}, Concurrency: {conc}, Attempts: {attempts}')
    print(f'  LLM mode: {llm_mode}, Profile: {profile_name}')
    print(f'{"="*60}')

    # 1. Start proxy
    print('\n[1/5] Starting replay proxy...')
    proxy_proc = start_proxy(cfg, llm_mode)
    if not proxy_proc:
        return 'ProxyFail'

    # 2. Start monitoring
    print('[2/5] Starting monitors...')
    mon_dir = output_dir / 'monitoring'
    mon_dir.mkdir(exist_ok=True)

    mpstat_proc = mpstat_sampler(mon_dir / 'mpstat.txt')
    vmstat_proc = vmstat_sampler(mon_dir / 'vmstat.txt')
    docker_proc = start_docker_monitor(mon_dir / 'docker_stats.txt')

    perf_cfg = PerfConfig(
        enabled=cfg.get('perf', {}).get('enabled', True),
        continuous_events=cfg.get('perf', {}).get('continuous_events', []),
        sample_interval_ms=cfg.get('perf', {}).get('sample_interval_ms', 100),
    )
    sampler = PerfSampler(perf_cfg)
    perf_ok = sampler.start_continuous(mon_dir)
    if perf_ok:
        print('  perf stat running (continuous)')
    else:
        print('  perf unavailable — using mpstat/vmstat only')

    # 3. Run Harbor
    print('[3/5] Running Harbor + TB2...')
    t0 = time.time()
    try:
        tail, elapsed = run_harbor(spec, proxy_port)
        print(f'  Completed in {elapsed:.0f}s')
        print(f'  Last output: {tail[-200:]}')
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f'  TIMEOUT after {elapsed:.0f}s')
        tail = 'TIMEOUT'
    except Exception as e:
        elapsed = time.time() - t0
        print(f'  ERROR: {e}')
        tail = str(e)

    # 4. Stop monitoring
    print('[4/5] Stopping monitors...')
    sampler.stop_continuous()
    for proc in [mpstat_proc, vmstat_proc, docker_proc]:
        if proc:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()

    stop_proxy(proxy_proc)

    # 5. Collect results
    print('[5/5] Collecting results...')
    phases = collect_task_phases(output_dir / 'jobs', output_dir / 'task_phases.json')
    write_stats(spec, elapsed, phases)

    # Stage attribution (if perf data available)
    perf_csv = mon_dir / 'perf_continuous.csv'
    if perf_csv.exists() and (output_dir / 'task_phases.json').exists():
        result = sampler.attribute_stages(
            output_dir / 'task_phases.json',
            perf_csv,
            output_dir / 'stage_attribution.json',
        )
        if result.get('summary', {}).get('ipc'):
            print(f'  IPC: {result["summary"]["ipc"]:.2f}')

    print(f'  Results: {output_dir}')
    return 'Success'


def main():
    p = argparse.ArgumentParser(description='AgentSysPerf local runner')
    p.add_argument('--config', type=Path, default=CONFIG_DIR / 'local.yml')
    p.add_argument('--llm-mode', choices=['off', 'replay', 'record', 'slm', 'live'])
    p.add_argument('--profile', default='base')
    p.add_argument('--tasks', type=int, default=None,
                   help='Run only first N tasks (for quick iteration)')
    args = p.parse_args()

    cfg = load_config(args.config)
    llm_mode = args.llm_mode or cfg.get('llm_mode', 'off')

    status = run_one(cfg, llm_mode, args.profile, args.tasks)
    print(f'\nFinal status: {status}')
    return 0 if status == 'Success' else 1


if __name__ == '__main__':
    sys.exit(main())
