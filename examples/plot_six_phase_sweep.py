#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Render Ring-1 charts from a 6-phase sweep_summary.json.

Produces three PNGs, each single-axis (no dual-axis — the #1 chart
mistake), using the validated categorical palette in fixed phase order:

  chart1_phase_stack.png   per-phase stacked latency, c=1 vs saturation
  chart3_agents_at_slo.png P99 loop latency vs concurrency + SLO line + knee
  chart5_throughput.png    throughput (loops/s) vs concurrency

Palette: reference categorical slots 1-6 (blue/aqua/yellow/green/violet/red),
validated colorblind-safe (worst adjacent CVD ΔE 24.2 on light surface).

Usage:
    python examples/plot_six_phase_sweep.py \
        --summary /tmp/agentsysperf_scratch/six_phase_sweep/sweep_summary.json \
        --out-dir /tmp/agentsysperf_scratch/six_phase_sweep/charts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import tempfile

# Scratch root for run artifacts. gettempdir() honours TMPDIR and falls
# back to "/tmp", so this resolves to the same paths as the hardcoded
# /tmp literals it replaced while no longer pinning output to a
# world-writable directory on systems that configure TMPDIR elsewhere.
_TMP = tempfile.gettempdir()

PHASE_ORDER = ["reason", "retrieve", "act", "admit", "context", "commit"]

# Validated categorical palette — fixed slot order, never cycled.
PHASE_COLOR = {
    "reason":   "#2a78d6",  # slot 1 blue
    "retrieve": "#1baf7a",  # slot 2 aqua
    "act":      "#eda100",  # slot 3 yellow
    "admit":    "#008300",  # slot 4 green
    "context":  "#4a3aa7",  # slot 5 violet
    "commit":   "#e34948",  # slot 6 red
}

# Chart chrome / ink (reference palette, light surface).
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
STATUS_CRITICAL = "#d03b3b"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK_PRIMARY,
})


def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def chart1_phase_stack(summary, out_dir):
    """Per-phase composition shift, baseline vs saturation.

    Absolute latencies differ ~280x between c=1 and c=50, so a linear
    stack hides the baseline entirely. The chart's job is the composition
    shift — which phase comes to dominate — so we normalize each bar to
    100% and annotate the absolute total per loop above it.
    """
    c_min = summary[0]
    c_sat = summary[-1]
    cells = [c_min, c_sat]
    totals = [
        sum((cell["per_phase_mean_ms"].get(p) or 0.0) for p in PHASE_ORDER)
        for cell in cells
    ]
    labels = [f"c={c_min['concurrency']}\n(baseline)",
              f"c={c_sat['concurrency']}\n(saturation)"]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    x = range(len(cells))
    bottoms = [0.0, 0.0]
    for phase in PHASE_ORDER:
        pct = [
            100.0 * (cells[i]["per_phase_mean_ms"].get(phase) or 0.0) / totals[i]
            if totals[i] else 0.0
            for i in range(len(cells))
        ]
        ax.bar(x, pct, bottom=bottoms, width=0.5,
               color=PHASE_COLOR[phase], label=phase, zorder=3,
               edgecolor=SURFACE, linewidth=2)  # 2px surface gap between segments
        for i, (v, b) in enumerate(zip(pct, bottoms)):
            if v >= 6.0:  # label only segments wide enough to read
                ax.text(i, b + v / 2, f"{v:.0f}%", ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold", zorder=4)
        bottoms = [b + v for b, v in zip(bottoms, pct)]

    # Absolute total per loop annotated above each bar (the magnitude story).
    for i, tot in enumerate(totals):
        ax.text(i, 101.5, f"{tot:.0f} ms/loop", ha="center", va="bottom",
                color=INK_PRIMARY, fontsize=10, fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, color=INK_SECONDARY)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Share of per-loop latency (%)")
    ax.set_title("Phase composition shift: baseline vs saturation",
                 color=INK_PRIMARY, fontweight="bold", loc="left")
    _style_axes(ax)
    ax.legend(frameon=False, loc="lower center", fontsize=9, ncol=3,
              labelcolor=INK_SECONDARY, bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    p = out_dir / "chart1_phase_stack.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def chart3_agents_at_slo(summary, out_dir):
    """P99 loop latency vs concurrency, with SLO line and agents-at-SLO knee."""
    cs = [c["concurrency"] for c in summary]
    p99 = [c["latency_ms"]["p99"] for c in summary]
    slo = summary[0]["slo_ms"]
    knee = max((c["concurrency"] for c in summary if c["p99_meets_slo"]), default=0)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(cs, p99, color=PHASE_COLOR["reason"], linewidth=2, marker="o",
            markersize=8, zorder=3, label="P99 loop latency")
    # SLO threshold as a status line (critical), icon+label, never color alone.
    ax.axhline(slo, color=STATUS_CRITICAL, linewidth=1.5, linestyle="--", zorder=2)
    ax.text(cs[-1], slo, f"  P99 SLO = {slo:.0f} ms", color=STATUS_CRITICAL,
            va="bottom", ha="right", fontsize=9, fontweight="bold")
    # Mark the knee — the delivered-agents-at-SLO number.
    if knee:
        ax.axvline(knee, color=MUTED, linewidth=1, linestyle=":", zorder=1)
        ax.text(knee, max(p99) * 0.9, f"knee: {knee} agents @ SLO",
                color=INK_PRIMARY, fontsize=9, fontweight="bold",
                ha="left", rotation=0)
    for cc, yy in zip(cs, p99):
        ax.text(cc, yy, f"  {yy:.0f}", color=INK_SECONDARY, fontsize=8,
                va="bottom", ha="left")

    ax.set_xlabel("Offered concurrency (agents in flight)")
    ax.set_ylabel("P99 loop latency (ms)")
    ax.set_title("Agents-at-SLO: where P99 breaks the latency target",
                 color=INK_PRIMARY, fontweight="bold", loc="left")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _style_axes(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    p = out_dir / "chart3_agents_at_slo.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def chart2_phase_growth(summary, out_dir):
    """Per-phase mean latency vs concurrency — the contention curves.

    Each phase is one line (fixed-order categorical color). The Admit line
    is the proof point: it grows steeply once concurrency exceeds the
    Router's parallel-request cap, showing admission control contends
    independently of per-core speed.
    """
    cs = [c["concurrency"] for c in summary]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for phase in PHASE_ORDER:
        ys = [c["per_phase_mean_ms"].get(phase) or 0.0 for c in summary]
        # Admit is the proof-point line — emphasize it; the rest recede.
        emphasize = phase == "admit"
        ax.plot(cs, ys, color=PHASE_COLOR[phase],
                linewidth=3 if emphasize else 1.8,
                marker="o", markersize=8 if emphasize else 6,
                zorder=4 if emphasize else 3, label=phase,
                alpha=1.0 if emphasize else 0.75)

    ax.set_xlabel("Offered concurrency (agents in flight)")
    ax.set_ylabel("Mean phase latency per loop (ms)")
    ax.set_yscale("log")  # phases span 3 orders of magnitude across the sweep
    ax.set_title("Per-phase latency growth under concurrency (admit gate emphasized)",
                 color=INK_PRIMARY, fontweight="bold", loc="left", fontsize=11)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _style_axes(ax)
    ax.grid(axis="both", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", fontsize=9, ncol=2,
              labelcolor=INK_SECONDARY)
    fig.tight_layout()
    p = out_dir / "chart2_phase_growth.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def chart5_throughput(summary, out_dir):
    """Throughput (loops/s) vs concurrency — single axis, its own chart."""
    cs = [c["concurrency"] for c in summary]
    thru = [c["throughput_loops_per_s"] for c in summary]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(cs, thru, color=PHASE_COLOR["retrieve"], linewidth=2, marker="o",
            markersize=8, zorder=3, label="Throughput")
    for cc, yy in zip(cs, thru):
        ax.text(cc, yy, f"  {yy:.1f}", color=INK_SECONDARY, fontsize=8,
                va="bottom", ha="left")
    ax.set_xlabel("Offered concurrency (agents in flight)")
    ax.set_ylabel("Throughput (loops / sec)")
    ax.set_title("Throughput vs concurrency",
                 color=INK_PRIMARY, fontweight="bold", loc="left")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _style_axes(ax)
    ax.legend(frameon=False, loc="upper right", fontsize=9, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    p = out_dir / "chart5_throughput.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", type=Path,
                    default=Path(f"{_TMP}/agentsysperf_scratch/six_phase_sweep/sweep_summary.json"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path(f"{_TMP}/agentsysperf_scratch/six_phase_sweep/charts"))
    args = ap.parse_args()

    with open(args.summary) as f:
        doc = json.load(f)
    # Accept both the badged summary ({headline, cells:[...]}) and a bare
    # list of cell rollups (partial-summary convenience files).
    summary = doc["cells"] if isinstance(doc, dict) and "cells" in doc else doc
    summary.sort(key=lambda c: c["concurrency"])
    args.out_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        chart1_phase_stack(summary, args.out_dir),
        chart2_phase_growth(summary, args.out_dir),
        chart3_agents_at_slo(summary, args.out_dir),
        chart5_throughput(summary, args.out_dir),
    ]
    for p in paths:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
