#!/usr/bin/env bash
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

# Density-scaling EMON sweep across ALL synthetic workload mixes.
#
# The `compile` mix was collected first.
# This runs the REMAINING mixes with the same methodology (2 cores/agent, 600
# turns, EMON + core-filtered pyEDP) so every workload class has a matching
# density-scaling TMA dataset.
#
# Each mix's full 8-density sweep takes ~45-55 min (8 x ~5-min compute + pyEDP).
# All 8 remaining mixes ≈ 6-7 hours. Run under nohup/tmux.
#
# Usage:
#   cd <repo> && PYTHONPATH=. bash experiments/scaling/run_all_mixes_density.sh
#   # or a subset:
#   MIXES="ml_train linalg" PYTHONPATH=. bash experiments/scaling/run_all_mixes_density.sh
#
# Env overrides: SEP_DIR, PYEDP_PYTHON (see density_emon_runner.py docstring),
#                DENSITIES, CORES_PER_AGENT, TURNS.

set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1   # repo root
export PYTHONPATH="${PYTHONPATH:-.}"

# All mixes EXCEPT compile (already done). 'mixed' rotates through the others.
MIXES="${MIXES:-ml_train linalg compress raytrace sat interpreter io_heavy mixed}"
DENSITIES="${DENSITIES:-1 2 4 8 12 16 24 32}"
CORES_PER_AGENT="${CORES_PER_AGENT:-2}"
TURNS="${TURNS:-600}"

echo "=============================================================="
echo " All-mixes density EMON sweep"
echo " mixes:     $MIXES"
echo " densities: $DENSITIES"
echo " config:    ${CORES_PER_AGENT}c/agent, $TURNS turns, EMON on"
echo "=============================================================="

for mix in $MIXES; do
    echo ""
    echo ">>> MIX: $mix  ($(date '+%H:%M:%S'))"
    python3 -m experiments.scaling.density_emon_runner \
        --mix "$mix" \
        --densities $DENSITIES \
        --cores-per-agent "$CORES_PER_AGENT" \
        --turns "$TURNS" \
        --output-dir "/tmp/agentsysperf_density_${mix}"
    rc=$?
    if [ $rc -ne 0 ]; then
        echo ">>> MIX $mix FAILED (rc=$rc) — continuing to next mix"
    fi
done

echo ""
echo "ALL MIXES DONE ($(date '+%H:%M:%S'))"
