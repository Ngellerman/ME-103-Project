#!/usr/bin/env python3
"""
Volume statistics and distributions for the OSB and L firebrand samples.

Reads the merged CSVs, then:
  1. Drops samples with no usable volume (blank, or volume exceeds convex hull)
  2. Prints averages and spread for each material, and writes them to a CSV
  3. Plots the volume distribution of each material on a shared log axis

Usage:
    python analyze_volumes.py
    python analyze_volumes.py --watertight-only   # also drop non-watertight meshes
    python analyze_volumes.py --out-dir results
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullLocator, ScalarFormatter
import numpy as np

MATERIALS = {
    "OSB": {"file": "OSB_volumes_masses.csv", "color": "#2a78d6"},
    "L": {"file": "L_volumes_masses.csv", "color": "#eb6834"},
}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"


def load_volumes(path: Path, watertight_only: bool) -> np.ndarray:
    """Return the usable volumes (cm^3) from one merged CSV."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    vols = []
    for r in rows:
        if not r["Volume (cm^3)"] or r["Volume Flag"].startswith("invalid"):
            continue
        if watertight_only and r["Watertight"] != "True":
            continue
        vols.append(float(r["Volume (cm^3)"]))
    return np.array(vols)


def summarize(material: str, vols: np.ndarray) -> dict:
    q1, median, q3 = np.percentile(vols, [25, 50, 75])
    return {
        "Material": material,
        "N": len(vols),
        "Mean (cm^3)": round(float(vols.mean()), 4),
        "Median (cm^3)": round(float(median), 4),
        "Std Dev (cm^3)": round(float(vols.std(ddof=1)), 4),
        "Min (cm^3)": round(float(vols.min()), 4),
        "Q1 (cm^3)": round(float(q1), 4),
        "Q3 (cm^3)": round(float(q3), 4),
        "Max (cm^3)": round(float(vols.max()), 4),
    }


def plot_distributions(data: dict, stats: dict, out: Path) -> None:
    # Volumes are strongly right-skewed, so use log-spaced bins shared by both panels.
    all_vols = np.concatenate(list(data.values()))
    bins = np.geomspace(all_vols.min(), all_vols.max(), 25)

    fig, axes = plt.subplots(len(data), 1, figsize=(9, 6.5), sharex=True, facecolor=SURFACE)
    for ax, (material, vols) in zip(axes, data.items()):
        ax.set_facecolor(SURFACE)
        ax.hist(vols, bins=bins, rwidth=0.88, color=MATERIALS[material]["color"], zorder=2)
        ax.set_xscale("log")
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
        ax.tick_params(axis="both", colors=INK_SECONDARY, length=0)
        ax.set_ylabel("Samples", color=INK_SECONDARY)
        ax.set_title(f"{material} (n = {len(vols)})", loc="left", color=INK, fontsize=12, fontweight="bold")

        mean = stats[material]["Mean (cm^3)"]
        median = stats[material]["Median (cm^3)"]
        ax.axvline(median, color=INK, linewidth=1.5, zorder=3)
        ax.axvline(mean, color=INK_SECONDARY, linewidth=1.5, zorder=3)
        top = ax.get_ylim()[1] * 1.12  # headroom so labels clear the tallest bar
        ax.set_ylim(0, top)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        label_bg = {"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5}
        ax.text(median * 0.97, top, f"Median {median:.2f}", ha="right", va="top", color=INK, fontsize=10, bbox=label_bg)
        ax.text(mean * 1.03, top, f"Mean {mean:.2f}", ha="left", va="top", color=INK_SECONDARY, fontsize=10, bbox=label_bg)

    axes[-1].set_xlabel("Volume (cm³, log scale)", color=INK_SECONDARY)
    axes[-1].set_xticks([0.3, 0.5, 1, 2, 5, 10, 15])
    axes[-1].xaxis.set_major_formatter(ScalarFormatter())
    axes[-1].xaxis.set_minor_locator(NullLocator())
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("."), help="folder containing the merged CSVs")
    ap.add_argument("--out-dir", type=Path, default=Path("."), help="folder for the summary CSV and plot")
    ap.add_argument("--watertight-only", action="store_true", help="exclude non-watertight meshes")
    args = ap.parse_args()

    data = {m: load_volumes(args.data_dir / c["file"], args.watertight_only) for m, c in MATERIALS.items()}
    stats = {m: summarize(m, v) for m, v in data.items()}

    fields = list(next(iter(stats.values())))
    print(("{:<9}{:>5}" + "{:>9}" * 7).format("Material", "N", "Mean", "Median", "StdDev", "Min", "Q1", "Q3", "Max"))
    for s in stats.values():
        print(("{:<9}{:>5}" + "{:>9.3f}" * 7).format(*s.values()))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "volume_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(stats.values())

    plot_path = args.out_dir / "volume_distribution.png"
    plot_distributions(data, stats, plot_path)
    print(f"\nWrote {summary_path} and {plot_path}")


if __name__ == "__main__":
    main()
