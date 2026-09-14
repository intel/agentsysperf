#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""
SLM Inference Interleaving Test

Tests whether SLM inference (Qwen2.5-1.5B BF16/AMX) can run alongside
tool-execution tasks without mutual degradation.

Test cases:
1. SLM solo (baseline) — pinned to node 1
2. SLM + path-tracing (FP-compute, fits L2) — both on node 1
3. SLM + train-fasttext (memory-BW, streams L3) — both on node 1

The SLM runs as a background Python process generating tokens continuously.
The TB2 task runs via Harbor concurrently on the same NUMA node.
perf stat -C monitors combined activity.
"""
import subprocess
import tempfile
import time
import os
import sys
import signal
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "harness"

HARBOR = os.environ.get("HARBOR", str(REPO / ".venv" / "bin" / "harbor"))
DATASET = os.environ.get("DATASET", str(HARNESS / "datasets" / "terminal-bench-2"))
COMPOSE = os.environ.get("COMPOSE", str(HARNESS / "configs" / "numa_node1.yml"))
OUTDIR = os.environ.get("OUTDIR", str(HARNESS / "results" / "slm_interleaving"))
CPUS = "43-85"
PERF_EVENTS = "cycles,instructions,cache-references,cache-misses,branch-instructions,branch-misses,L1-dcache-load-misses,LLC-load-misses,context-switches,page-faults"

SLM_SCRIPT = """
import torch
import time
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = 'Qwen/Qwen2.5-1.5B'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
model.eval()
torch.set_num_threads(43)

prompt = 'Analyze the performance characteristics of modern CPU architectures:'
inputs = tokenizer(prompt, return_tensors='pt')

# Warmup
with torch.no_grad():
    model.generate(**inputs, max_new_tokens=10)

# Run until killed
print('SLM_READY', flush=True)
total_tokens = 0
t0 = time.time()
while True:
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    total_tokens += out.shape[1] - inputs['input_ids'].shape[1]
    elapsed = time.time() - t0
    if int(elapsed) % 10 == 0:
        print(f'SLM: {total_tokens} tokens in {elapsed:.0f}s = {total_tokens/elapsed:.1f} tok/s', flush=True)
"""


def _clean_containers():
    """Remove leftover compose containers from a previous test.

    Was `docker ps -q | xargs -r docker rm -f` under a shell. Splitting the
    pipeline into two Python steps drops the shell entirely and, as a bonus,
    fixes the silent case where `xargs -r` swallowed a docker failure: the ids
    now come back as data we can see.
    """
    ids = subprocess.run(
        ["docker", "ps", "-a", "--filter",
         "label=com.docker.compose.project", "-q"],
        capture_output=True, text=True,
    ).stdout.split()
    if ids:
        subprocess.run(["docker", "rm", "-f", *ids], capture_output=True)


def run_test(label, task=None):
    """Run SLM with optional concurrent TB2 task."""
    print(f"\n{'='*60}")
    print(f"  Test: {label}")
    print(f"{'='*60}")

    outpath = os.path.join(OUTDIR, label)
    os.makedirs(outpath, exist_ok=True)

    _clean_containers()

    # Write SLM script to temp file
    slm_script_path = os.path.join(tempfile.gettempdir(), "slm_continuous.py")
    with open(slm_script_path, "w") as f:
        f.write(SLM_SCRIPT)

    # Start perf stat. Exec'd directly rather than via a shell: with shell=True
    # perf_proc.pid was /bin/sh's, so the SIGTERM below hit the shell and perf
    # could miss its chance to flush the counter summary into perf.txt.
    perf_proc = subprocess.Popen(
        ["perf", "stat", "-C", CPUS, "-e", PERF_EVENTS, "--", "sleep", "120"],
        stdout=open(f"{outpath}/perf.txt", "w"),
        stderr=subprocess.STDOUT)
    print(f"  perf PID: {perf_proc.pid}")

    # Start SLM (pinned to node 1 via taskset)
    slm_proc = subprocess.Popen(
        ["taskset", "-c", "43-85", "python3", slm_script_path],
        stdout=open(f"{outpath}/slm_output.txt", "w"),
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid)
    print(f"  SLM PID: {slm_proc.pid}")

    # Wait for SLM to be ready
    time.sleep(15)  # Model loading takes ~5s
    print("  SLM running.")

    harbor_proc = None
    if task:
        # Start TB2 task concurrently
        harbor_cmd = [HARBOR, "run", "-p", f"{DATASET}/{task}", "-a", "oracle",
                      "-n", "1", "--yes", "--extra-docker-compose", str(COMPOSE)]
        harbor_proc = subprocess.Popen(
            harbor_cmd,
            stdout=open(f"{outpath}/harbor_stdout.txt", "w"),
            stderr=open(f"{outpath}/harbor_stderr.txt", "w"))
        print(f"  Harbor ({task}) PID: {harbor_proc.pid}")

    # Let them run together
    duration = 60 if task != "train-fasttext" else 90
    print(f"  Running for {duration}s...")
    time.sleep(duration)

    # Stop SLM
    os.killpg(os.getpgid(slm_proc.pid), signal.SIGTERM)
    slm_proc.wait()
    print("  SLM stopped.")

    # Wait for Harbor if running
    if harbor_proc:
        harbor_proc.wait()
        print(f"  Harbor exit: {harbor_proc.returncode}")

    # Stop perf
    perf_proc.send_signal(signal.SIGTERM)
    perf_proc.wait()
    print(f"  Perf saved: {outpath}/perf.txt")

    _clean_containers()

    # Parse SLM throughput
    try:
        with open(f"{outpath}/slm_output.txt") as f:
            lines = [l for l in f.readlines() if "tok/s" in l]
            if lines:
                print(f"  SLM final: {lines[-1].strip()}")
    except:
        pass

    print(f"  Done: {label}")


if __name__ == "__main__":
    print("SLM Interleaving Test")
    print(f"Date: {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    print(f"Model: Qwen2.5-1.5B (BF16, 43 threads)")
    print(f"Node: NUMA 1 (CPUs 43-85)")

    # Test 1: SLM solo
    run_test("slm_solo")

    # Test 2: SLM + path-tracing (FP-compute, should not contend)
    run_test("slm+path-tracing", task="path-tracing")

    # Test 3: SLM + train-fasttext (memory-BW, likely contends)
    run_test("slm+train-fasttext", task="train-fasttext")

    print("\n" + "="*60)
    print("  All interleaving tests complete")
    print("="*60)
