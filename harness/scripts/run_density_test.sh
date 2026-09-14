#!/bin/bash
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

# Multi-agent density test: 4 → 8 → 16 concurrent agents on NUMA node 1
# Uses Terminus-2 with replay proxy (365-turn fixture, flexible mode)
# All containers pinned to node 1 (CPUs 43-85) via compose overlay
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HARNESS="$REPO/harness"
HARBOR="${HARBOR:-$REPO/.venv/bin/harbor}"
DATASET="${DATASET:-$HARNESS/datasets/terminal-bench-2}"
OUTDIR="${OUTDIR:-$HARNESS/results/density_test}"
COMPOSE="${COMPOSE:-$HARNESS/configs/numa_node1.yml}"
FIXTURE="${FIXTURE:-$REPO//path/to/your/fixture.jsonl}"
PROXY_PORT=4001
CPUS="43-85"

PERF_EVENTS="cycles,instructions,cache-references,cache-misses,branch-instructions,branch-misses,L1-dcache-load-misses,LLC-load-misses,context-switches,page-faults"

cd "$HARNESS"

start_proxy() {
    echo "Starting replay proxy (flexible mode, trial=ab4183d383a978b2)..."
    source .venv/bin/activate
    python scripts/replay_proxy.py \
        --mode replay \
        --fixture "$FIXTURE" \
        --replay-trial ab4183d383a978b2 \
        --no-strict \
        --port $PROXY_PORT \
        > "$OUTDIR/proxy.log" 2>&1 &
    PROXY_PID=$!
    echo "Proxy PID: $PROXY_PID"

    # Wait for proxy to be ready
    for i in $(seq 1 15); do
        if curl -s "http://localhost:$PROXY_PORT/healthz" > /dev/null 2>&1; then
            echo "Proxy ready."
            return 0
        fi
        sleep 1
    done
    echo "ERROR: Proxy failed to start"
    return 1
}

stop_proxy() {
    if [ -n "${PROXY_PID:-}" ]; then
        kill $PROXY_PID 2>/dev/null || true
        wait $PROXY_PID 2>/dev/null || true
        echo "Proxy stopped."
    fi
}

run_density() {
    local N=$1
    local LABEL="n${N}"
    local DURATION=90  # perf monitoring window

    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "║  Density Test: $N concurrent agents          ║"
    echo "╚══════════════════════════════════════════════╝"
    echo "CPUs: $CPUS (NUMA node 1)"
    echo "Timeout: 60s agent timeout"
    echo ""

    mkdir -p "$OUTDIR/$LABEL"

    # Clean leftovers
    docker ps -a --filter "label=com.docker.compose.project" -q | xargs -r docker rm -f 2>/dev/null || true

    # Start perf stat monitoring node 1 CPUs
    perf stat -C "$CPUS" \
        -e "$PERF_EVENTS" \
        -- sleep $DURATION \
        > "$OUTDIR/$LABEL/perf.txt" 2>&1 &
    PERF_PID=$!
    echo "perf stat PID: $PERF_PID"

    # Also capture vmstat for context-switch rate over time
    vmstat 1 $DURATION > "$OUTDIR/$LABEL/vmstat.txt" 2>&1 &
    VMSTAT_PID=$!

    # Run Harbor with N concurrent agents
    # Use dataset path with --n-tasks to limit, and -n for concurrency
    # Terminus-2 runs LLM calls from host (not container), so localhost works
    OPENAI_API_KEY="sk-proj-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" \
    $HARBOR run \
        -p "$DATASET" \
        -a terminus-2 \
        -m "openai/gpt-4" \
        -n "$N" \
        --n-tasks "$N" \
        --yes \
        --extra-docker-compose "$COMPOSE" \
        --agent-timeout-multiplier 0.05 \
        --ak "api_base=http://localhost:$PROXY_PORT/v1" \
        > "$OUTDIR/$LABEL/harbor_stdout.txt" 2> "$OUTDIR/$LABEL/harbor_stderr.txt"
    HARBOR_EXIT=$?
    echo "Harbor exit: $HARBOR_EXIT"

    # Stop monitoring
    kill -INT $PERF_PID 2>/dev/null || true
    kill $VMSTAT_PID 2>/dev/null || true
    wait $PERF_PID 2>/dev/null || true
    wait $VMSTAT_PID 2>/dev/null || true

    # Count how many containers actually ran
    echo "Containers observed:" >> "$OUTDIR/$LABEL/summary.txt"
    cat "$OUTDIR/$LABEL/harbor_stdout.txt" | grep -c "trial\|Trial\|TRIAL" >> "$OUTDIR/$LABEL/summary.txt" 2>/dev/null || true

    # Capture docker stats snapshot (if containers still running)
    docker stats --no-stream --format "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" 2>/dev/null \
        > "$OUTDIR/$LABEL/docker_stats.txt" || true

    # Cleanup
    docker ps -a --filter "label=com.docker.compose.project" -q | xargs -r docker rm -f 2>/dev/null || true

    echo "Results saved to $OUTDIR/$LABEL/"
    echo ""
}

echo "Multi-Agent Density Test"
echo "Date: $(date -Iseconds)"
echo "Hardware: Xeon 6787P, NUMA node 1 (CPUs 43-85)"
echo ""

# Start replay proxy
start_proxy
trap stop_proxy EXIT

# Run density tests: 4, 8, 16 concurrent agents
run_density 4
run_density 8
run_density 16

echo "╔══════════════════════════════════════════════╗"
echo "║  All density tests complete                  ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "Results:"
for d in "$OUTDIR"/n*/perf.txt; do
    echo "  $d"
done
