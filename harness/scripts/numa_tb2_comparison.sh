#!/bin/bash
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

# NUMA comparison for train-fasttext on nodes 0, 1, 2
# Pins Docker container to specific NUMA node via cpuset, measures with perf cgroup
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HARNESS="$REPO/harness"
HARBOR="${HARBOR:-$REPO/.venv/bin/harbor}"
TASK="${TASK:-$HARNESS/datasets/terminal-bench-2/train-fasttext}"
OUTDIR="${OUTDIR:-$HARNESS/results/numa_comparison_tb2}"

# NUMA node CPU lists (physical cores only, no HT for cleaner measurement)
NODE0_CPUS="0-42"
NODE1_CPUS="43-85"
NODE2_CPUS="86-128"

PERF_EVENTS="cycles,instructions,cache-references,cache-misses,branch-instructions,branch-misses,L1-dcache-load-misses,LLC-load-misses,context-switches,page-faults"

run_on_node() {
    local NODE=$1
    local CPUS=$2
    local LABEL="node${NODE}"

    echo "=========================================="
    echo "  Running train-fasttext on NUMA node $NODE"
    echo "  CPUs: $CPUS"
    echo "=========================================="

    mkdir -p "$OUTDIR/$LABEL"

    # Clean any leftover containers
    docker ps -a --filter "name=harbor" -q | xargs -r docker rm -f 2>/dev/null || true

    # Run Harbor with NUMA pinning via Docker runtime options
    # Harbor uses --run-args to pass extra Docker options
    echo "Starting Harbor with cpuset-cpus=$CPUS, cpuset-mems=$NODE..."

    $HARBOR run \
        -p "$TASK" \
        -a oracle -n 1 --yes \
        --run-args "--cpuset-cpus=$CPUS --cpuset-mems=$NODE" \
        > "$OUTDIR/$LABEL/harbor_stdout.txt" 2> "$OUTDIR/$LABEL/harbor_stderr.txt" &
    HARBOR_PID=$!

    # Wait for the Docker container to appear
    echo "Waiting for container..."
    CONTAINER_ID=""
    for i in $(seq 1 30); do
        CONTAINER_ID=$(docker ps --filter "name=harbor" --format "{{.ID}}" 2>/dev/null | head -1)
        if [ -n "$CONTAINER_ID" ]; then
            break
        fi
        sleep 1
    done

    if [ -z "$CONTAINER_ID" ]; then
        echo "ERROR: No harbor container appeared after 30s"
        wait $HARBOR_PID 2>/dev/null || true
        return 1
    fi

    echo "Container: $CONTAINER_ID"

    # Get cgroup path
    CGROUP_PATH=$(docker inspect "$CONTAINER_ID" --format '{{.State.Pid}}' 2>/dev/null)
    DOCKER_CGROUP="system.slice/docker-${CONTAINER_ID}*.scope"
    # Get the full container ID for cgroup path
    FULL_ID=$(docker inspect "$CONTAINER_ID" --format '{{.Id}}')
    CGROUP="system.slice/docker-${FULL_ID}.scope"

    echo "Cgroup: $CGROUP"

    # Start perf stat on the container cgroup
    sudo perf stat -a --cgroup "$CGROUP" \
        -e "$PERF_EVENTS" \
        -- sleep 300 \
        > "$OUTDIR/$LABEL/perf_cgroup.txt" 2>&1 &
    PERF_PID=$!

    echo "perf PID: $PERF_PID, waiting for Harbor to complete..."

    # Wait for Harbor to finish
    wait $HARBOR_PID 2>/dev/null
    HARBOR_EXIT=$?
    echo "Harbor exited with code $HARBOR_EXIT"

    # Give a few seconds for trailing activity, then stop perf
    sleep 3
    sudo kill -INT $PERF_PID 2>/dev/null || true
    wait $PERF_PID 2>/dev/null || true

    echo "Perf data saved to $OUTDIR/$LABEL/perf_cgroup.txt"

    # Verify the container was actually pinned
    docker inspect "$CONTAINER_ID" --format 'CpusetCpus={{.HostConfig.CpusetCpus}} CpusetMems={{.HostConfig.CpusetMems}}' \
        > "$OUTDIR/$LABEL/cpuset_verify.txt" 2>/dev/null || echo "container already removed" > "$OUTDIR/$LABEL/cpuset_verify.txt"

    # Clean up
    docker rm -f "$CONTAINER_ID" 2>/dev/null || true

    echo "Node $NODE complete."
    echo ""
}

echo "NUMA TB2 Comparison: train-fasttext"
echo "Date: $(date -Iseconds)"
echo ""

# Run on each node sequentially
run_on_node 0 "$NODE0_CPUS"
run_on_node 1 "$NODE1_CPUS"
run_on_node 2 "$NODE2_CPUS"

echo "=========================================="
echo "All runs complete. Results in $OUTDIR/"
echo "=========================================="
