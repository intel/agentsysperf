#!/usr/bin/env python3
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""
Synthetic task workloads that approximate the CPU signatures of TB2 tasks.

These run directly (no Harbor/Docker) to validate the perf instrumentation
pipeline and produce baseline per-task-type hardware fingerprints.

Each task exercises a different microarchitectural axis:
  - compile_like: branch-heavy, instruction-stream-bound (gcc-like)
  - ml_train_like: vectorized FP, memory-BW (fasttext/numpy)
  - video_like: streaming memory access, SIMD (transcode)
  - linalg_like: dense matrix ops, memory-BW, potential AMX
  - io_like: large file reads, page faults (ETL)
  - compress_like: cache-friendly, high IPC (LZ-style)
  - raytrace_like: FP-heavy, random memory (path tracing)
  - sat_like: branch-heavy, random memory (constraint solving)
  - interpreter_like: unpredictable branches (eval loop)
  - control_like: short, minimal work (agent orchestration)

Usage:
    python synthetic_tasks.py                  # run all, ~30s total
    python synthetic_tasks.py --task compile   # run one
    python synthetic_tasks.py --perf           # run with perf stat per task
"""
import argparse
import array
import ast
import hashlib
import math
import os
import random
import struct
import subprocess
import sys
import time
from pathlib import Path
import tempfile

# Scratch root for run artifacts. gettempdir() honours TMPDIR and falls
# back to "/tmp", so this resolves to the same paths as the hardcoded
# /tmp literals it replaced while no longer pinning output to a
# world-writable directory on systems that configure TMPDIR elsewhere.
_TMP = tempfile.gettempdir()

HARNESS_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = HARNESS_ROOT / 'results' / 'synthetic_baseline'


def compile_like(duration_s: float = 3.0):
    """Branch-heavy string processing simulating compiler-like workload.
    High branch rate, moderate IPC, instruction-stream-bound."""
    end = time.time() + duration_s
    # Simulate lexing/parsing: lots of conditionals on character classes
    data = "int main() { int x = 0; for(int i=0; i<100; i++) { x += i * 2; } return x; }\n" * 10000
    count = 0
    while time.time() < end:
        tokens = []
        i = 0
        while i < len(data):
            c = data[i]
            if c.isalpha() or c == '_':
                j = i + 1
                while j < len(data) and (data[j].isalnum() or data[j] == '_'):
                    j += 1
                tokens.append(data[i:j])
                i = j
            elif c.isdigit():
                j = i + 1
                while j < len(data) and data[j].isdigit():
                    j += 1
                tokens.append(int(data[i:j]))
                i = j
            else:
                i += 1
        count += 1
    return {'iterations': count, 'tokens_per_iter': len(tokens)}


def ml_train_like(duration_s: float = 3.0):
    """Vectorized floating-point with memory streaming.
    High memory bandwidth demand, moderate IPC, FP-heavy."""
    end = time.time() + duration_s
    n = 10000
    # Simulate matrix-vector operations (dot products, gradient updates)
    weights = array.array('d', [random.gauss(0, 0.01) for _ in range(n)])
    gradient = array.array('d', [0.0] * n)
    inputs = array.array('d', [random.gauss(0, 1) for _ in range(n)])
    lr = 0.001
    count = 0
    while time.time() < end:
        # Forward: dot product
        activation = sum(w * x for w, x in zip(weights, inputs))
        # Loss gradient
        error = activation - 1.0
        # Backward: gradient computation + weight update
        for i in range(n):
            gradient[i] = error * inputs[i]
            weights[i] -= lr * gradient[i]
        count += 1
    return {'iterations': count, 'vector_size': n}


def linalg_like(duration_s: float = 3.0):
    """Dense matrix multiplication — memory-BW-bound, AMX-candidate.
    Large working set, streaming access pattern."""
    end = time.time() + duration_s
    n = 200  # 200x200 matrix → 320KB per matrix (doubles)
    A = [[random.random() for _ in range(n)] for _ in range(n)]
    B = [[random.random() for _ in range(n)] for _ in range(n)]
    count = 0
    while time.time() < end:
        # Naive matmul (cache-unfriendly on purpose to stress memory)
        C = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                s = 0.0
                for k in range(n):
                    s += A[i][k] * B[k][j]
                C[i][j] = s
        count += 1
    return {'iterations': count, 'matrix_size': n}


def io_like(duration_s: float = 3.0):
    """Large sequential file reads — IO-bound, page-fault-heavy."""
    end = time.time() + duration_s
    # Create a temp file to read
    tmp = Path(f'{_TMP}/agentsysperf_io_test.bin')
    if not tmp.exists():
        tmp.write_bytes(os.urandom(64 * 1024 * 1024))  # 64MB
    count = 0
    bytes_read = 0
    while time.time() < end:
        with open(tmp, 'rb') as f:
            while True:
                chunk = f.read(4096)
                if not chunk:
                    break
                bytes_read += len(chunk)
        count += 1
    return {'iterations': count, 'bytes_read': bytes_read}


def compress_like(duration_s: float = 3.0):
    """LZ-style compression — cache-friendly, high IPC, hash-chain lookups."""
    end = time.time() + duration_s
    # Simulate hash-chain dictionary matching
    data = bytes(random.getrandbits(8) for _ in range(100000))
    # Add some repetition for realistic compression
    data = data + data[:50000] + data[25000:75000]
    count = 0
    while time.time() < end:
        # Simple hash-chain simulation
        window_size = 32768
        hash_table = {}
        matches = 0
        for i in range(3, len(data)):
            key = data[i-3:i]
            if key in hash_table:
                pos = hash_table[key]
                if i - pos < window_size:
                    matches += 1
            hash_table[key] = i
        count += 1
    return {'iterations': count, 'match_rate': matches / len(data)}


def raytrace_like(duration_s: float = 3.0):
    """FP-heavy with random memory access — path tracing approximation.
    High FP throughput demand, random access to scene data."""
    end = time.time() + duration_s
    # Simulate ray-sphere intersection tests
    spheres = [(random.uniform(-10, 10), random.uniform(-10, 10),
                random.uniform(-10, 10), random.uniform(0.5, 2.0))
               for _ in range(200)]
    count = 0
    hits = 0
    while time.time() < end:
        for _ in range(1000):
            # Random ray
            ox, oy, oz = 0, 0, -20
            dx = random.uniform(-1, 1)
            dy = random.uniform(-1, 1)
            dz = 1.0
            norm = math.sqrt(dx*dx + dy*dy + dz*dz)
            dx, dy, dz = dx/norm, dy/norm, dz/norm
            # Test against all spheres
            for sx, sy, sz, sr in spheres:
                lx, ly, lz = sx-ox, sy-oy, sz-oz
                tca = lx*dx + ly*dy + lz*dz
                if tca < 0:
                    continue
                d2 = lx*lx + ly*ly + lz*lz - tca*tca
                if d2 < sr*sr:
                    hits += 1
                    break
        count += 1
    return {'iterations': count, 'rays': count * 1000, 'hits': hits}


def sat_like(duration_s: float = 3.0):
    """Branch-heavy with random memory — constraint satisfaction.
    Low IPC due to branch mispredictions and random access."""
    end = time.time() + duration_s
    n_vars = 100
    n_clauses = 400
    # Generate random 3-SAT instance
    clauses = []
    for _ in range(n_clauses):
        clause = []
        for _ in range(3):
            var = random.randint(0, n_vars-1)
            neg = random.choice([True, False])
            clause.append((var, neg))
        clauses.append(clause)

    count = 0
    while time.time() < end:
        # Random restart local search
        assignment = [random.choice([True, False]) for _ in range(n_vars)]
        for _ in range(1000):
            # Find unsatisfied clause
            unsat = None
            for clause in clauses:
                satisfied = False
                for var, neg in clause:
                    val = assignment[var] ^ neg
                    if val:
                        satisfied = True
                        break
                if not satisfied:
                    unsat = clause
                    break
            if unsat is None:
                break
            # Flip a random variable in the unsatisfied clause
            var, _ = random.choice(unsat)
            assignment[var] = not assignment[var]
        count += 1
    return {'iterations': count}


def interpreter_like(duration_s: float = 3.0):
    """Unpredictable branches — bytecode interpreter simulation.
    Very low IPC due to indirect dispatch."""
    end = time.time() + duration_s
    # Simulate a stack-based VM
    ops = [random.randint(0, 9) for _ in range(10000)]
    count = 0
    while time.time() < end:
        stack = []
        pc = 0
        for _ in range(len(ops)):
            op = ops[pc % len(ops)]
            pc += 1
            if op == 0:    # push
                stack.append(pc)
            elif op == 1:  # pop
                if stack: stack.pop()
            elif op == 2:  # add
                if len(stack) >= 2:
                    b, a = stack.pop(), stack.pop()
                    stack.append(a + b)
            elif op == 3:  # sub
                if len(stack) >= 2:
                    b, a = stack.pop(), stack.pop()
                    stack.append(a - b)
            elif op == 4:  # dup
                if stack: stack.append(stack[-1])
            elif op == 5:  # swap
                if len(stack) >= 2:
                    stack[-1], stack[-2] = stack[-2], stack[-1]
            elif op == 6:  # jmp
                pc = (pc + 7) % len(ops)
            elif op == 7:  # jz
                if stack and stack[-1] == 0:
                    pc = (pc + 3) % len(ops)
            elif op == 8:  # mul
                if len(stack) >= 2:
                    b, a = stack.pop(), stack.pop()
                    stack.append((a * b) & 0xFFFFFFFF)
            elif op == 9:  # nop
                pass
        count += 1
    return {'iterations': count}


def control_like(duration_s: float = 0.5):
    """Minimal work — simulates agent control plane overhead.
    Very short, low CPU, mostly setup/teardown."""
    end = time.time() + duration_s
    count = 0
    while time.time() < end:
        # Simulate JSON parsing + HTTP-like overhead
        data = '{"role": "assistant", "content": "executing step 3"}'
        for _ in range(100):
            _ = hash(data)
            _ = len(data.encode())
        count += 1
    return {'iterations': count}


TASKS = {
    'compile': ('Compile-like (branch-heavy, lexing)', compile_like),
    'ml_train': ('ML training (vectorized FP, mem-BW)', ml_train_like),
    'linalg': ('Linear algebra (dense matmul, AMX-candidate)', linalg_like),
    'io': ('ETL/IO (sequential file reads)', io_like),
    'compress': ('Compression (cache-friendly, high IPC)', compress_like),
    'raytrace': ('Ray tracing (FP-heavy, random access)', raytrace_like),
    'sat': ('SAT solving (branch-heavy, random mem)', sat_like),
    'interpreter': ('Interpreter (unpredictable dispatch)', interpreter_like),
    'control': ('Control plane (minimal, short)', control_like),
}


def run_with_perf(name: str, func, duration_s: float, output_dir: Path) -> dict:
    """Run a task under perf stat and capture counters."""
    output_dir.mkdir(parents=True, exist_ok=True)
    perf_out = output_dir / f'{name}_perf.txt'

    events = [
        'cycles', 'instructions', 'cache-references', 'cache-misses',
        'branch-instructions', 'branch-misses', 'L1-dcache-load-misses',
        'LLC-load-misses', 'LLC-store-misses',
        'context-switches', 'cpu-migrations', 'page-faults',
    ]

    # Run the task in a subprocess so perf can attach to it
    task_script = f"""
import sys; sys.path.insert(0, '{HARNESS_ROOT / "scripts"}')
from synthetic_tasks import {func.__name__}
result = {func.__name__}({duration_s})
import json; print(json.dumps(result))
"""
    cmd = [
        'perf', 'stat', '-e', ','.join(events), '-x', ',',
        '--', sys.executable, '-c', task_script,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration_s + 30)

    # Parse perf output (on stderr, CSV format)
    counters = {}
    for line in result.stderr.splitlines():
        parts = line.strip().split(',')
        if len(parts) >= 4:
            try:
                val = int(parts[0]) if parts[0].strip() != '<not counted>' else 0
                event = parts[2] if len(parts) > 2 else parts[3]
                # Try different column positions
                for p in parts[2:]:
                    if any(k in p for k in ['cycles', 'instructions', 'cache', 'branch',
                                            'LLC', 'context', 'cpu-', 'page']):
                        event = p.strip()
                        break
                counters[event] = val
            except (ValueError, IndexError):
                continue

    # Parse task output. The task prints a dict literal on its last line, so
    # literal_eval covers it — and unlike eval it cannot execute code if a
    # workload ever emits something unexpected on that line.
    task_result = {}
    try:
        task_result = ast.literal_eval(result.stdout.strip().split('\n')[-1])
    except (ValueError, SyntaxError, IndexError):
        pass

    # Compute derived metrics
    cycles = counters.get('cycles', 0)
    instructions = counters.get('instructions', 0)
    cache_refs = counters.get('cache-references', 0)
    cache_misses = counters.get('cache-misses', 0)
    branch_instr = counters.get('branch-instructions', 0)
    branch_misses = counters.get('branch-misses', 0)

    derived = {
        'ipc': instructions / cycles if cycles else 0,
        'cache_miss_rate': cache_misses / cache_refs if cache_refs else 0,
        'branch_miss_rate': branch_misses / branch_instr if branch_instr else 0,
        'branch_pct': branch_instr / instructions if instructions else 0,
    }

    # Save raw perf output
    perf_out.write_text(result.stderr)

    return {
        'task': name,
        'duration_s': duration_s,
        'counters': counters,
        'derived': derived,
        'task_result': task_result,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=list(TASKS.keys()), default=None)
    p.add_argument('--perf', action='store_true', help='Run with perf stat')
    p.add_argument('--duration', type=float, default=3.0)
    args = p.parse_args()

    output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks_to_run = {args.task: TASKS[args.task]} if args.task else TASKS

    results = []
    print(f"{'Task':<14} {'Description':<42} {'IPC':>5} {'Cache Miss%':>11} {'Branch Miss%':>12}")
    print("-" * 90)

    for name, (desc, func) in tasks_to_run.items():
        if args.perf:
            r = run_with_perf(name, func, args.duration, output_dir)
            d = r['derived']
            print(f"{name:<14} {desc:<42} {d['ipc']:>5.2f} {d['cache_miss_rate']*100:>9.1f}% {d['branch_miss_rate']*100:>10.1f}%")
            results.append(r)
        else:
            t0 = time.time()
            res = func(args.duration)
            elapsed = time.time() - t0
            print(f"{name:<14} {desc:<42} {elapsed:.1f}s  {res}")

    if args.perf and results:
        import json
        summary_path = output_dir / 'fingerprints.json'
        summary_path.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nResults saved to: {summary_path}")


if __name__ == '__main__':
    main()
