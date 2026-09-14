#!/bin/bash
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

# =============================================================================
# AgentSysPerf Full AWS Experiment Launcher
# =============================================================================
# Usage:
#   tmux new -s agentsysperf
#   bash experiments/scaling/run_full_aws.sh
#
# Runs the complete density scaling matrix unattended (~12 hr per platform).
# Safe to detach (Ctrl-B D) and reconnect (tmux attach -t agentsysperf).
#
# Output: /tmp/agentsysperf_aws/<timestamp>/
# =============================================================================

set -euo pipefail

# Repo root, derived from this script's own location rather than guessed at.
# A $HOME/agentsysperf default is wrong for any checkout that isn't there, and
# the failure lands on `python -m experiments.scaling...` as a confusing
# ModuleNotFoundError rather than on the cd. $AGENTSYSPERF_REPO still wins.
REPO_ROOT="${AGENTSYSPERF_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PLATFORM=$(uname -m)
OUTPUT_BASE="/tmp/agentsysperf_aws/${PLATFORM}_${TIMESTAMP}"
LOG="${OUTPUT_BASE}/experiment.log"
TURNS=1444  # ~13 min per config at 0.54s/turn

mkdir -p "${OUTPUT_BASE}"

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "${LOG}"
}

run_phase() {
    local phase_name="$1"
    local output_dir="${OUTPUT_BASE}/$2"
    shift 2
    local args="$*"

    log "━━━ PHASE: ${phase_name} ━━━"
    log "  Output: ${output_dir}"
    log "  Args: ${args}"

    cd "${REPO_ROOT}"

    python -m experiments.scaling.run_experiment \
        ${args} \
        --turns ${TURNS} \
        --output "${output_dir}" \
        --quiet \
        2>&1 | tee -a "${LOG}"

    log "  Done: ${phase_name}"
    log ""
}

# =============================================================================
log "============================================================"
log "  AGENTSYSPERF FULL AWS EXPERIMENT"
log "  Platform: ${PLATFORM}"
log "  Output:   ${OUTPUT_BASE}"
log "  Started:  $(date)"
log "============================================================"
log ""

# System info
log "System: $(lscpu | grep 'Model name' | head -1 | awk -F: '{print $2}' | xargs)"
log "Cores:  $(nproc)"
log "L3:     $(cat /sys/devices/system/cpu/cpu0/cache/index3/size 2>/dev/null || echo 'unknown')"
log "Memory: $(free -h | awk '/Mem:/ {print $2}')"
log "NUMA:   $(numactl --hardware 2>/dev/null | head -1 || echo 'unavailable')"
log ""

# =============================================================================
# PHASE 1: Baseline (single agent, all mixes)
# ~40 min
# =============================================================================
run_phase "Baseline (compile)" "01_baseline" \
    --densities 1 --placements spread --mixes compile --cores-per-agent 4

run_phase "Baseline (ml_train)" "01_baseline" \
    --densities 1 --placements spread --mixes ml_train --cores-per-agent 4

run_phase "Baseline (io_heavy)" "01_baseline" \
    --densities 1 --placements spread --mixes io_heavy --cores-per-agent 4

run_phase "Baseline (raytrace)" "01_baseline" \
    --densities 1 --placements spread --mixes raytrace --cores-per-agent 4

run_phase "Baseline (sat)" "01_baseline" \
    --densities 1 --placements spread --mixes sat --cores-per-agent 4

run_phase "Baseline (interpreter)" "01_baseline" \
    --densities 1 --placements spread --mixes interpreter --cores-per-agent 4

# =============================================================================
# PHASE 2: Density Scaling — spread placement, compile
# ~90 min (7 densities × 13 min)
# =============================================================================
run_phase "Density spread (compile, 2-core)" "02_density_spread" \
    --densities 1,4,8,16,24,32,46 --placements spread --mixes compile --cores-per-agent 2

# =============================================================================
# PHASE 3: Density Scaling — intra_node placement
# ~52 min (4 densities × 13 min)
# =============================================================================
run_phase "Density intra_node (compile, 4-core)" "03_density_intra" \
    --densities 1,4,8,16 --placements intra_node --mixes compile --cores-per-agent 4

run_phase "Density intra_node (compile, 2-core)" "03_density_intra" \
    --densities 1,4,8,16 --placements intra_node --mixes compile --cores-per-agent 2

# =============================================================================
# PHASE 4: Workload Mixes at Key Densities
# ~4.3 hr (5 mixes × 4 densities × 13 min = 260 min)
# =============================================================================
for mix in ml_train io_heavy raytrace sat interpreter; do
    run_phase "Mix: ${mix} (spread)" "04_mixes" \
        --densities 1,8,16,32 --placements spread --mixes ${mix} --cores-per-agent 2
done

# =============================================================================
# PHASE 5: Mixed Fleet (heterogeneous agents)
# ~52 min (4 densities × 13 min)
# =============================================================================
run_phase "Mixed fleet (spread)" "05_mixed_fleet" \
    --densities 1,4,8,16,32 --placements spread --mixes mixed --cores-per-agent 2

# =============================================================================
# PHASE 6: Cross-node placement (if NUMA available)
# ~52 min (4 densities × 13 min)
# =============================================================================
if numactl --hardware 2>/dev/null | grep -q "node 1"; then
    log "NUMA detected — running cross_node placement"
    run_phase "Density cross_node (compile)" "06_cross_node" \
        --densities 1,4,8,16 --placements cross_node --mixes compile --cores-per-agent 2
else
    log "Single NUMA node — skipping cross_node placement"
fi

# =============================================================================
# PHASE 7: High-density stress test (push to limits)
# ~26 min (2 densities × 13 min)
# =============================================================================
run_phase "High density stress" "07_stress" \
    --densities 46,64 --placements spread --mixes compile --cores-per-agent 1

# =============================================================================
# DONE — Package results
# =============================================================================
log ""
log "============================================================"
log "  EXPERIMENT COMPLETE"
log "  Platform: ${PLATFORM}"
log "  Output:   ${OUTPUT_BASE}"
log "  Finished: $(date)"
log "============================================================"

# Combine all results into one file
python3 -c "
import json
from pathlib import Path

all_results = []
base = Path('${OUTPUT_BASE}')
for f in sorted(base.rglob('all_results.json')):
    if f.parent == base:
        continue
    data = json.loads(f.read_text())
    all_results.extend(data)

combined = base / 'all_results_combined.json'
combined.write_text(json.dumps(all_results, indent=2))
print(f'Combined {len(all_results)} configs into {combined}')
"

# Create tarball for easy transfer
cd /tmp/agentsysperf_aws
tar czf "${PLATFORM}_${TIMESTAMP}.tar.gz" "${PLATFORM}_${TIMESTAMP}/"
log "Tarball: /tmp/agentsysperf_aws/${PLATFORM}_${TIMESTAMP}.tar.gz"
log "Size: $(du -sh /tmp/agentsysperf_aws/${PLATFORM}_${TIMESTAMP}.tar.gz | cut -f1)"

log ""
log "To copy results:"
log "  scp <this-host>:/tmp/agentsysperf_aws/${PLATFORM}_${TIMESTAMP}.tar.gz ."
log ""
log "Done."
