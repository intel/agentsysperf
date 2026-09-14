#!/bin/bash
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

# Run train-fasttext on NUMA nodes 0, 1, 2 with perf cgroup profiling
# Each run: Harbor starts container pinned to node's CPUs, perf stat attaches to cgroup
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HARNESS="$REPO/harness"
HARBOR="${HARBOR:-$REPO/.venv/bin/harbor}"
TASK_PATH="datasets/terminal-bench-2/train-fasttext"
OUTDIR="results/numa_comparison_tb2"
CONFIGS="configs"

PERF_EVENTS="cycles,instructions,cache-references,cache-misses,branch-instructions,branch-misses,L1-dcache-load-misses,LLC-load-misses,context-switches,page-faults"

# CPU ranges per NUMA node (physical cores only)
declare -A NODE_CPUS
NODE_CPUS[0]="0-42"
NODE_CPUS[1]="43-85"
NODE_CPUS[2]="86-128"

cd "$HARNESS"

run_node() {
    local NODE=$1
    local LABEL="node${NODE}"
    local COMPOSE="${CONFIGS}/numa_node${NODE}.yml"
    local CPUS="${NODE_CPUS[$NODE]}"

    echo ""
    echo "╔══════════════════════════════════════════╗"
    echo "║  NUMA Node $NODE: train-fasttext         ║"
    echo "╚══════════════════════════════════════════╝"
    echo "Compose overlay: $COMPOSE"
    echo "CPUs: $CPUS"
    echo "Output: $OUTDIR/$LABEL/"
    echo ""

    mkdir -p "$OUTDIR/$LABEL"

    # Clean leftovers
    docker ps -a --filter "label=com.docker.compose.project" -q | xargs -r docker rm -f 2>/dev/null || true

    # Start Harbor in background
    $HARBOR run \
        -p "$TASK_PATH" \
        -a oracle -n 1 --yes \
        --extra-docker-compose "$COMPOSE" \
        > "$OUTDIR/$LABEL/harbor_stdout.txt" 2> "$OUTDIR/$LABEL/harbor_stderr.txt" &
    HARBOR_PID=$!

    # Wait for container
    echo "Waiting for container to start..."
    CONTAINER_ID=""
    for i in $(seq 1 60); do
        CONTAINER_ID=$(docker ps --filter "label=com.docker.compose.project" --format "{{.ID}}" 2>/dev/null | head -1)
        if [ -n "$CONTAINER_ID" ]; then
            break
        fi
        sleep 1
    done

    if [ -z "$CONTAINER_ID" ]; then
        echo "ERROR: No container appeared after 60s"
        wait $HARBOR_PID 2>/dev/null || true
        return 1
    fi

    echo "Container: $CONTAINER_ID"
    docker inspect "$CONTAINER_ID" --format 'CpusetCpus={{.HostConfig.CpusetCpus}}' | tee "$OUTDIR/$LABEL/cpuset_verify.txt"

    # Start perf stat monitoring the pinned CPUs (no sudo needed)
    # Since the container is pinned to these CPUs exclusively, this captures
    # only the container's activity (plus minimal kernel overhead on those cores)
    perf stat -C "$CPUS" \
        -e "$PERF_EVENTS" \
        -- sleep 300 \
        > "$OUTDIR/$LABEL/perf_cgroup.txt" 2>&1 &
    PERF_PID=$!
    echo "perf stat PID: $PERF_PID (monitoring CPUs $CPUS)"

    # Wait for Harbor to complete
    echo "Waiting for Harbor (train-fasttext takes ~261s)..."
    wait $HARBOR_PID 2>/dev/null
    HARBOR_EXIT=$?
    echo "Harbor exit: $HARBOR_EXIT"

    # Let trailing IO settle, then stop perf
    sleep 5
    kill -INT $PERF_PID 2>/dev/null || true
    wait $PERF_PID 2>/dev/null || true

    echo "Perf saved: $OUTDIR/$LABEL/perf_cgroup.txt"

    # Cleanup container
    docker rm -f "$CONTAINER_ID" 2>/dev/null || true

    echo "Node $NODE complete."
}

echo "NUMA Comparison: train-fasttext on nodes 0, 1, 2"
echo "Date: $(date -Iseconds)"
echo "Hardware: Xeon 6787P, 4 NUMA nodes × 43 cores"
echo ""

run_node 0
run_node 1
run_node 2

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  All 3 NUMA runs complete               ║"
echo "╚══════════════════════════════════════════╝"
echo "Results in: $OUTDIR/"
ls -la "$OUTDIR"/node*/perf_cgroup.txt 2>/dev/null
