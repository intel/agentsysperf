#!/usr/bin/env python3
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Run the parallel agent density scaling experiment.

Usage:
    # Quick smoke test (2 densities, 1 placement, 1 mix):
    python -m experiments.scaling.run_experiment --quick

    # Medium run (all densities, 1 placement, 1 mix):
    python -m experiments.scaling.run_experiment --medium

    # Full matrix (all densities x placements x mixes):
    python -m experiments.scaling.run_experiment --full

    # Custom:
    python -m experiments.scaling.run_experiment --densities 1,4,8,16 --placements spread --mixes compute_heavy
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.scaling.config import (
    ExperimentConfig,
    ExperimentMatrix,
    PhaseMix,
    Placement,
)
from experiments.scaling.orchestrator import ExperimentResult, run_experiment
import tempfile

# Scratch root for run artifacts. gettempdir() honours TMPDIR and falls
# back to "/tmp", so this resolves to the same paths as the hardcoded
# /tmp literals it replaced while no longer pinning output to a
# world-writable directory on systems that configure TMPDIR elsewhere.
_TMP = tempfile.gettempdir()


OUTPUT_BASE = Path(f"{_TMP}/agentsysperf_scaling")


def run_matrix(
    matrix: ExperimentMatrix,
    output_dir: Path,
    verbose: bool = True,
) -> list[ExperimentResult]:
    """Run all configurations in the experiment matrix."""
    configs = matrix.configs()

    print("=" * 70)
    print("  PARALLEL AGENT DENSITY SCALING EXPERIMENT")
    print("  Platform: Granite Rapids Xeon (96C/192T, 480MB L3, SNC3)")
    print("=" * 70)
    print(f"\n  Configurations: {len(configs)}")
    print(f"  Densities:      {matrix.densities}")
    print(f"  Placements:     {[p.value for p in matrix.placements]}")
    print(f"  Phase mixes:    {[m.value for m in matrix.phase_mixes]}")
    print(f"  Cores/agent:    {matrix.cores_per_agent}")
    print(f"  Output:         {output_dir}")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    start_time = time.time()

    for i, config in enumerate(configs, 1):
        print(f"\n{'─'*70}")
        print(f"  [{i}/{len(configs)}] Running...")
        try:
            result = run_experiment(config, output_dir, verbose=verbose)
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append(ExperimentResult(
                config=config, agent_results=[], error=str(e),
            ))

    elapsed = time.time() - start_time

    # Save combined results
    combined_file = output_dir / "all_results.json"
    combined_file.write_text(json.dumps(
        [r.to_dict() for r in results],
        indent=2, default=str,
    ))

    # Print summary
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Configs run: {len(results)}")
    print(f"  Successful: {sum(1 for r in results if r.error is None)}")
    print(f"  Results: {combined_file}")

    # Print scaling summary table
    print(f"\n  {'Density':<8} {'Placement':<12} {'Mix':<15} {'Throughput':>12} {'MeanTurn':>10}")
    print(f"  {'─'*8} {'─'*12} {'─'*15} {'─'*12} {'─'*10}")
    for r in results:
        if r.error:
            continue
        print(f"  {r.config.density:<8} {r.config.placement.value:<12} "
              f"{r.config.phase_mix.value:<15} "
              f"{r.mean_throughput():>10.2f}/s {r.agent_results[0].mean_turn_ms() if r.agent_results else 0:>8.0f}ms")

    print(f"\n{'='*70}\n")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run parallel agent density scaling experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true",
                     help="Smoke test: densities [1,4], intra_node, compile")
    mode.add_argument("--medium", action="store_true",
                     help="Medium: all densities, intra_node, compile")
    mode.add_argument("--full", action="store_true",
                     help="Full matrix: all densities x placements x mixes")

    parser.add_argument("--densities", type=str, default=None,
                       help="Comma-separated density levels (e.g., '1,4,8,16')")
    parser.add_argument("--placements", type=str, default=None,
                       help="Comma-separated placements (intra_node,cross_node,spread)")
    parser.add_argument("--mixes", type=str, default=None,
                       help="Comma-separated phase mixes (compile,ml_train,linalg,compress,raytrace,sat,interpreter,io_heavy,mixed)")
    parser.add_argument("--cores-per-agent", type=int, default=4)
    parser.add_argument("--turns", type=int, default=8)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    # Determine output dir
    output_dir = Path(args.output) if args.output else OUTPUT_BASE / f"run_{int(time.time())}"

    # Build matrix based on mode
    if args.quick:
        matrix = ExperimentMatrix(
            densities=[1, 4],
            placements=[Placement.INTRA_NODE],
            phase_mixes=[PhaseMix.COMPILE],
            cores_per_agent=args.cores_per_agent,
            turns=args.turns,
        )
    elif args.medium:
        matrix = ExperimentMatrix(
            densities=[1, 2, 4, 8, 12, 16, 24],
            placements=[Placement.INTRA_NODE],
            phase_mixes=[PhaseMix.COMPILE],
            cores_per_agent=args.cores_per_agent,
            turns=args.turns,
        )
    elif args.full:
        matrix = ExperimentMatrix(
            cores_per_agent=args.cores_per_agent,
            turns=args.turns,
        )
    else:
        # Custom from flags
        densities = [int(d) for d in args.densities.split(",")] if args.densities else [1, 2, 4, 8]
        placements = (
            [Placement(p.strip()) for p in args.placements.split(",")]
            if args.placements else [Placement.INTRA_NODE]
        )
        mixes = (
            [PhaseMix(m.strip()) for m in args.mixes.split(",")]
            if args.mixes else [PhaseMix.COMPILE]
        )
        matrix = ExperimentMatrix(
            densities=densities,
            placements=placements,
            phase_mixes=mixes,
            cores_per_agent=args.cores_per_agent,
            turns=args.turns,
        )

    run_matrix(matrix, output_dir, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
