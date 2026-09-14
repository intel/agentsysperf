#!/bin/bash
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

# AgentSysPerf Local Harness — Setup Script
# Installs dependencies needed to run the harness on this server.
#
# Prerequisites: Docker (already available), Python 3.12+ (already available)
#
# What this does:
#   1. Install Python packages (fastapi, uvicorn, httpx, pyyaml)
#   2. Install system tools (numactl, sysstat for mpstat)
#   3. Install Harbor + Terminus-2 (the agent + orchestrator)
#   4. Pull Terminal-Bench 2 task images
#   5. Verify perf access
#
# Run: bash scripts/setup_local.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== AgentSysPerf Local Harness Setup ==="
echo "Server: $(hostname)"
echo "CPU: $(lscpu | grep 'Model name' | sed 's/.*: *//')"
echo "Cores: $(nproc)"
echo ""

# 1. Python packages
echo "[1/5] Installing Python packages..."
pip3 install --user fastapi uvicorn httpx pyyaml 2>/dev/null || \
    pip install --user fastapi uvicorn httpx pyyaml
echo "  Done."

# 2. System tools
echo ""
echo "[2/5] Checking system tools..."
if ! command -v mpstat &>/dev/null; then
    echo "  mpstat not found. Install with: sudo apt install sysstat"
else
    echo "  mpstat: OK"
fi
if ! command -v numactl &>/dev/null; then
    echo "  numactl not found. Install with: sudo apt install numactl"
else
    echo "  numactl: OK"
fi
if ! command -v perf &>/dev/null; then
    echo "  perf not found. Install with: sudo apt install linux-tools-$(uname -r)"
else
    echo "  perf: OK ($(perf --version))"
fi

# 3. Harbor + Terminus-2
echo ""
echo "[3/5] Harbor + Terminus-2..."
if command -v harbor &>/dev/null; then
    echo "  harbor: OK ($(harbor --version 2>/dev/null || echo 'installed'))"
else
    echo "  Harbor not installed."
    echo "  Install from: https://github.com/laude-institute/harbor"
    echo "  Quick: pip install --user harbor-ai"
    echo "  Or:    git clone https://github.com/laude-institute/harbor && cd harbor && pip install -e ."
    echo ""
    echo "  After installing Harbor, install Terminus-2:"
    echo "    harbor install terminus-2"
fi

# 4. Terminal-Bench 2
echo ""
echo "[4/5] Terminal-Bench 2 task images..."
if command -v harbor &>/dev/null; then
    echo "  Checking if TB2 dataset is available..."
    harbor list-datasets 2>/dev/null | grep -q terminal-bench && \
        echo "  terminal-bench: OK" || \
        echo "  Install with: harbor install terminal-bench/terminal-bench-2"
else
    echo "  (skipped — install Harbor first)"
fi

# 5. Perf access
echo ""
echo "[5/5] Perf counter access..."
PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid)
echo "  perf_event_paranoid = $PARANOID"
if [ "$PARANOID" -gt 1 ]; then
    echo "  WARNING: perf counters are restricted (paranoid=$PARANOID)"
    echo "  To unlock hardware counters, run:"
    echo "    sudo sysctl kernel.perf_event_paranoid=-1"
    echo ""
    echo "  The harness will still work without perf (uses mpstat/vmstat)"
    echo "  but per-stage IPC/cache characterization requires perf access."
else
    echo "  OK — perf counters accessible"
    # Quick test
    perf stat -e cycles,instructions -- sleep 0.01 2>/dev/null && \
        echo "  Verified: perf stat works" || \
        echo "  WARNING: perf stat test failed"
fi

echo ""
echo "=== Setup Summary ==="
echo "Harness root: $HARNESS_ROOT"
echo "Config: $HARNESS_ROOT/config/local.yml"
echo ""
echo "To run (after Harbor + TB2 are installed):"
echo "  cd $HARNESS_ROOT"
echo "  python scripts/runner.py --llm-mode off --tasks 3"
echo ""
echo "To run replay proxy standalone (for testing):"
echo "  python scripts/replay_proxy.py --mode off --port 4001"
