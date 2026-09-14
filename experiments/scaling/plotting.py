#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Generate scaling experiment charts.

Produces:
  A. Scaling curves (throughput, IPC, LLC MPKI, completion time vs density)
  B. TMA heatmaps (per mix x placement)
  C. NUMA comparison bar charts
  D. Phase breakdown stacked area

Usage:
    python -m experiments.scaling.plotting /tmp/agentsysperf_scaling/run_<ts>
    python -m experiments.scaling.plotting /tmp/agentsysperf_scaling/run_<ts> --format png --dpi 150
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.ticker import MaxNLocator
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False


PLACEMENT_COLORS = {
    "intra_node": "#d32f2f",
    "cross_node": "#1976d2",
    "spread": "#388e3c",
}

MIX_COLORS = {
    "compute_heavy": "#d32f2f",
    "io_heavy": "#1976d2",
    "balanced": "#ff9800",
    "mixed": "#9c27b0",
}

MIX_MARKERS = {
    "compute_heavy": "o",
    "io_heavy": "s",
    "balanced": "^",
    "mixed": "D",
}


def _ensure_deps():
    if not HAS_MPL:
        print("ERROR: matplotlib required. Install with: pip install matplotlib", file=sys.stderr)
        sys.exit(1)


def plot_throughput_vs_density(
    curves: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
    fmt: str = "png",
    dpi: int = 150,
) -> Path:
    """Plot A1: Normalized per-agent throughput vs density, grouped by mix."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for curve_key, points in sorted(curves.items()):
        parts = curve_key.split("_", 1)
        placement = parts[0] if len(parts) > 1 else curve_key
        mix = parts[1] if len(parts) > 1 else ""
        # Use full key for label parsing
        label_parts = curve_key.rsplit("_", 1)

        densities = [p["density"] for p in points]
        normalized = [p["normalized_throughput"] for p in points]
        aggregate = [p["aggregate_throughput"] for p in points]

        # Determine color by mix
        color = None
        for mk, mc in MIX_COLORS.items():
            if mk in curve_key:
                color = mc
                break
        linestyle = "--" if "cross_node" in curve_key else "-." if "spread" in curve_key else "-"

        ax1.plot(densities, normalized, marker="o", markersize=5,
                 label=curve_key, color=color, linestyle=linestyle)
        ax2.plot(densities, aggregate, marker="s", markersize=5,
                 label=curve_key, color=color, linestyle=linestyle)

    ax1.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
    ax1.axhline(y=0.7, color="red", linestyle=":", alpha=0.3, label="70% threshold")
    ax1.set_xlabel("Agent Density (N)")
    ax1.set_ylabel("Normalized Per-Agent Throughput")
    ax1.set_title("Per-Agent Throughput Degradation")
    ax1.legend(fontsize=7, loc="lower left")
    ax1.set_ylim(0, 1.1)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Agent Density (N)")
    ax2.set_ylabel("Aggregate Throughput (turns/s)")
    ax2.set_title("System-Wide Throughput")
    ax2.legend(fontsize=7, loc="upper left")
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_dir / f"throughput_vs_density.{fmt}"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_ipc_vs_density(
    curves: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
    fmt: str = "png",
    dpi: int = 150,
) -> Path:
    """Plot A2: IPC vs density, grouped by placement."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for curve_key, points in sorted(curves.items()):
        densities = [p["density"] for p in points]
        ipcs = [p["ipc"] for p in points]

        color = None
        for pk, pc in PLACEMENT_COLORS.items():
            if pk in curve_key:
                color = pc
                break
        marker = "o"
        for mk, mm in MIX_MARKERS.items():
            if mk in curve_key:
                marker = mm
                break

        ax.plot(densities, ipcs, marker=marker, markersize=6,
                label=curve_key, color=color)

    ax.set_xlabel("Agent Density (N)")
    ax.set_ylabel("Instructions Per Cycle (IPC)")
    ax.set_title("IPC Degradation Under Contention")
    ax.legend(fontsize=7, loc="lower left")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_dir / f"ipc_vs_density.{fmt}"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_llc_mpki_vs_density(
    curves: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
    fmt: str = "png",
    dpi: int = 150,
) -> Path:
    """Plot A3: LLC MPKI vs density (log-scale), inflection annotated."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for curve_key, points in sorted(curves.items()):
        densities = [p["density"] for p in points]
        mpki = [p["llc_mpki"] for p in points]

        if all(m == 0 for m in mpki):
            continue

        color = None
        for mk, mc in MIX_COLORS.items():
            if mk in curve_key:
                color = mc
                break

        ax.semilogy(densities, [max(m, 0.01) for m in mpki], marker="o",
                    markersize=6, label=curve_key, color=color)

    # 2x baseline reference line
    ax.set_xlabel("Agent Density (N)")
    ax.set_ylabel("LLC MPKI (log scale)")
    ax.set_title("L3 Cache Pressure vs Density")
    ax.legend(fontsize=7, loc="upper left")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    out = output_dir / f"llc_mpki_vs_density.{fmt}"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_latency_vs_density(
    curves: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
    fmt: str = "png",
    dpi: int = 150,
) -> Path:
    """Plot A4: Turn latency (p50 + p95) vs density."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for curve_key, points in sorted(curves.items()):
        densities = [p["density"] for p in points]
        p50 = [p["p50_ms"] for p in points]
        p95 = [p["p95_ms"] for p in points]

        color = None
        for mk, mc in MIX_COLORS.items():
            if mk in curve_key:
                color = mc
                break

        ax.plot(densities, p50, marker="o", markersize=5,
                label=f"{curve_key} (p50)", color=color, linestyle="-")
        ax.plot(densities, p95, marker="v", markersize=4,
                label=f"{curve_key} (p95)", color=color, linestyle="--", alpha=0.7)

    ax.set_xlabel("Agent Density (N)")
    ax.set_ylabel("Turn Latency (ms)")
    ax.set_title("Turn Completion Time (p50 + p95)")
    ax.legend(fontsize=6, loc="upper left", ncol=2)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_dir / f"latency_vs_density.{fmt}"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_numa_comparison(
    numa_data: List[Dict[str, Any]],
    output_dir: Path,
    fmt: str = "png",
    dpi: int = 150,
) -> Optional[Path]:
    """Plot C: NUMA placement comparison bar chart."""
    if not numa_data:
        return None

    placements = ["intra_node", "cross_node", "spread"]
    densities = sorted(set(d["density"] for d in numa_data))

    if not densities:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.25
    x_positions = list(range(len(densities)))

    for i, placement in enumerate(placements):
        throughputs = []
        for density in densities:
            entry = next((d for d in numa_data if d["density"] == density), None)
            if entry and placement in entry:
                throughputs.append(entry[placement]["throughput"])
            else:
                throughputs.append(0)

        offset = (i - 1) * width
        bars = ax.bar([x + offset for x in x_positions], throughputs, width,
                      label=placement, color=PLACEMENT_COLORS.get(placement, "#607d8b"))

    ax.set_xlabel("Agent Density")
    ax.set_ylabel("Per-Agent Throughput (turns/s)")
    ax.set_title("NUMA Placement Effect on Throughput")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(d) for d in densities])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = output_dir / f"numa_comparison.{fmt}"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_phase_breakdown(
    phase_data: List[Dict[str, Any]],
    output_dir: Path,
    fmt: str = "png",
    dpi: int = 150,
) -> Optional[Path]:
    """Plot D: Phase breakdown stacked area chart."""
    if not phase_data:
        return None

    # Group by (placement, mix), plot one chart per group
    from collections import defaultdict
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in phase_data:
        groups[f"{p['placement']}_{p['phase_mix']}"].append(p)

    n_groups = len(groups)
    if n_groups == 0:
        return None

    cols = min(n_groups, 3)
    rows = (n_groups + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for idx, (group_key, points) in enumerate(sorted(groups.items())):
        ax = axes[idx // cols][idx % cols]
        points.sort(key=lambda p: p["density"])

        densities = [p["density"] for p in points]
        reason = [p["reason_pct"] for p in points]
        act = [p["act_pct"] for p in points]
        overhead = [p["overhead_pct"] for p in points]

        ax.stackplot(densities, reason, act, overhead,
                     labels=["Reason", "Act", "Overhead"],
                     colors=["#f44336", "#4caf50", "#9e9e9e"],
                     alpha=0.8)
        ax.set_xlabel("Density")
        ax.set_ylabel("% Wall-Clock")
        ax.set_title(group_key, fontsize=9)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=7, loc="lower right")

    # Hide empty subplots
    for idx in range(n_groups, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.suptitle("Phase Time Breakdown vs Density", fontsize=12)
    plt.tight_layout()
    out = output_dir / f"phase_breakdown.{fmt}"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_tma_heatmap(
    tma_data: List[Dict[str, Any]],
    output_dir: Path,
    fmt: str = "png",
    dpi: int = 150,
) -> Optional[Path]:
    """Plot B: TMA heatmap (density x TMA category)."""
    if not tma_data or not HAS_NP:
        return None

    tma_keys = ["Frontend_Bound", "Backend_Bound", "Bad_Speculation", "Retiring",
                "Memory_Bound", "Core_Bound"]

    # Filter to keys that actually have data
    available_keys = [k for k in tma_keys if any(k in d for d in tma_data)]
    if not available_keys:
        return None

    densities = sorted(set(d["density"] for d in tma_data))
    matrix = np.zeros((len(densities), len(available_keys)))

    for i, density in enumerate(densities):
        entries = [d for d in tma_data if d["density"] == density]
        if entries:
            entry = entries[0]
            for j, key in enumerate(available_keys):
                matrix[i, j] = entry.get(key, 0)

    fig, ax = plt.subplots(figsize=(8, max(4, len(densities) * 0.5)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=100)

    ax.set_xticks(range(len(available_keys)))
    ax.set_xticklabels([k.replace("_", "\n") for k in available_keys], fontsize=8)
    ax.set_yticks(range(len(densities)))
    ax.set_yticklabels([str(d) for d in densities])
    ax.set_xlabel("TMA Category")
    ax.set_ylabel("Agent Density")
    ax.set_title("TMA Breakdown vs Density")

    # Annotate cells
    for i in range(len(densities)):
        for j in range(len(available_keys)):
            val = matrix[i, j]
            if val > 0:
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=8, color="white" if val > 50 else "black")

    fig.colorbar(im, ax=ax, label="% Pipeline Slots")
    plt.tight_layout()
    out = output_dir / f"tma_heatmap.{fmt}"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_all_plots(
    results_dir: Path,
    output_dir: Optional[Path] = None,
    fmt: str = "png",
    dpi: int = 150,
) -> List[Path]:
    """Generate all plots from analysis report.

    Args:
        results_dir: Directory with analysis_report.json (from analysis.py)
        output_dir: Where to save plots (default: results_dir/plots/)
        fmt: Image format (png, pdf, svg)
        dpi: Resolution

    Returns:
        List of generated plot file paths.
    """
    _ensure_deps()

    report_file = results_dir / "analysis_report.json"
    if not report_file.exists():
        # Try running analysis first
        from .analysis import analyze, generate_report
        analysis = analyze(results_dir)
        report = generate_report(analysis)
        report_file.write_text(json.dumps(report, indent=2, default=str))
    else:
        report = json.loads(report_file.read_text())

    if output_dir is None:
        output_dir = results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    curves = report.get("scaling_curves", {})
    numa = report.get("numa_comparison", [])
    phases = report.get("phase_breakdown", [])
    tma = report.get("tma_heatmap")

    generated = []

    if curves:
        generated.append(plot_throughput_vs_density(curves, output_dir, fmt, dpi))
        generated.append(plot_ipc_vs_density(curves, output_dir, fmt, dpi))
        generated.append(plot_llc_mpki_vs_density(curves, output_dir, fmt, dpi))
        generated.append(plot_latency_vs_density(curves, output_dir, fmt, dpi))

    if numa:
        path = plot_numa_comparison(numa, output_dir, fmt, dpi)
        if path:
            generated.append(path)

    if phases:
        path = plot_phase_breakdown(phases, output_dir, fmt, dpi)
        if path:
            generated.append(path)

    if tma:
        path = plot_tma_heatmap(tma, output_dir, fmt, dpi)
        if path:
            generated.append(path)

    return generated


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate scaling experiment plots")
    parser.add_argument("results_dir", type=Path, help="Path to experiment results directory")
    parser.add_argument("--output", "-o", type=Path, default=None,
                       help="Output directory for plots (default: <results_dir>/plots/)")
    parser.add_argument("--format", "-f", choices=["png", "pdf", "svg"], default="png")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    plots = generate_all_plots(args.results_dir, args.output, args.format, args.dpi)
    print(f"Generated {len(plots)} plots in {args.output or args.results_dir / 'plots'}:")
    for p in plots:
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
