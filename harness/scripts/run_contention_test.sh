#!/bin/bash
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

# Multi-task contention: run two different TB2 tasks on same NUMA node simultaneously
# Measures interference between different hardware profiles
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HARNESS="$REPO/harness"
HARBOR="${HARBOR:-$REPO/.venv/bin/harbor}"
DATASET="${DATASET:-$HARNESS/datasets/terminal-bench-2}"
OUTDIR="${OUTDIR:-$HARNESS/results/contention_tb2}"
COMPOSE="${COMPOSE:-$HARNESS/configs/numa_node1.yml}"
CPUS="43-85"

PERF_EVENTS="cycles,instructions,cache-references,cache-misses,branch-instructions,branch-misses,L1-dcache-load-misses,LLC-load-misses,context-switches,page-faults"

cd "$HARNESS"

run_pair() {
    local TASK_A=$1
    local TASK_B=$2
    local LABEL="${TASK_A}+${TASK_B}"
    local DURATION=300  # long enough for train-fasttext (261s)

    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "║  Contention: $TASK_A + $TASK_B"
    echo "╚══════════════════════════════════════════════╝"
    echo "Both pinned to NUMA node 1 (CPUs $CPUS)"
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

    # Start both tasks simultaneously (separate job dirs to avoid lock conflict)
    # Task A
    $HARBOR run \
        -p "$DATASET/$TASK_A" \
        -a oracle -n 1 --yes \
        --extra-docker-compose "$COMPOSE" \
        --job-name "contention_${TASK_A}" \
        -o "$OUTDIR/$LABEL/jobs_a" \
        > "$OUTDIR/$LABEL/harbor_${TASK_A}.txt" 2>&1 &
    PID_A=$!
    echo "Task A ($TASK_A) PID: $PID_A"

    # Small delay to avoid timestamp collision
    sleep 2

    # Task B
    $HARBOR run \
        -p "$DATASET/$TASK_B" \
        -a oracle -n 1 --yes \
        --extra-docker-compose "$COMPOSE" \
        --job-name "contention_${TASK_B}" \
        -o "$OUTDIR/$LABEL/jobs_b" \
        > "$OUTDIR/$LABEL/harbor_${TASK_B}.txt" 2>&1 &
    PID_B=$!
    echo "Task B ($TASK_B) PID: $PID_B"

    # Wait for both to complete
    echo "Waiting for both tasks..."
    wait $PID_A 2>/dev/null
    EXIT_A=$?
    echo "  $TASK_A exit: $EXIT_A"

    wait $PID_B 2>/dev/null
    EXIT_B=$?
    echo "  $TASK_B exit: $EXIT_B"

    # Stop perf
    sleep 3
    kill -INT $PERF_PID 2>/dev/null || true
    wait $PERF_PID 2>/dev/null || true

    echo "Perf saved: $OUTDIR/$LABEL/perf.txt"

    # Cleanup
    docker ps -a --filter "label=com.docker.compose.project" -q | xargs -r docker rm -f 2>/dev/null || true

    echo "Pair complete: $LABEL"
}

echo "Multi-Task Contention Test"
echo "Date: $(date -Iseconds)"
echo "Hardware: Xeon 6787P, NUMA node 1 (CPUs 43-85)"
echo ""
echo "Testing interference between different hardware profiles:"
echo "  1. train-fasttext + path-tracing (mem-BW vs FP)"
echo "  2. train-fasttext + build-cython-ext (mem-BW vs L1d)"
echo "  3. path-tracing + constraints-scheduling (FP vs branch)"
echo ""

mkdir -p "$OUTDIR"

# Run the three contention pairs
run_pair "train-fasttext" "path-tracing"
run_pair "train-fasttext" "build-cython-ext"
run_pair "path-tracing" "constraints-scheduling"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  All contention pairs complete               ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "Results:"
for d in "$OUTDIR"/*/perf.txt; do
    echo "  $d"
done
