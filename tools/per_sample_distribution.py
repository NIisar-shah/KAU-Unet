"""Per-sample distribution figure: box + strip plot of per-test-image
Dice and HD95 across the 300 held-out test images, comparing the baseline
U-Net and KAN-UNet.

For each seed's run directory we read ``test_per_sample.csv`` (long-format,
one row per patient x view x phase x class). For each test sample we
average the metric across the five seeds, yielding a 5-seed-mean per-
sample distribution. This is then box+strip-plotted side-by-side by
model, by class.

Layout: 2 rows (Dice top, HD95 bottom) x 4 cols
        (LV endocardium | LV myocardium | left atrium | foreground mean).

A paired (per-sample) Wilcoxon signed-rank test is computed per panel and
the p-value annotated above each pair, with star coding (*, **, ***).

Usage:
    python tools/per_sample_distribution.py \
        --runs_root ./runs \
        --out_dir   runs/aggregate
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


RUN_DIR_RE = re.compile(r"^(?P<model>.+)_seed(?P<seed>-?\d+)$")

# Colour and label conventions, kept consistent with training_curves and
# qualitative figures.
COLOURS = {
    "baseline_unet": "#1f77b4",
    "kan_unet":      "#d62728",
}
LABELS = {
    "baseline_unet": "Baseline\nU-Net",
    "kan_unet":      "KAN-UNet\n(ours)",
}
PRETTY_CLASS = {
    "LV_endocardium": "LV endocardium",
    "LV_myocardium":  "LV myocardium",
    "left_atrium":    "Left atrium",
    "_foreground":    "Foreground mean",
}


def load_per_sample_csvs(runs_root: Path) -> Dict[str, pd.DataFrame]:
    """Find every test_per_sample.csv under runs_root, group by model."""
    by_model: Dict[str, List[pd.DataFrame]] = defaultdict(list)
    for d in sorted(runs_root.iterdir()):
        if not d.is_dir():
            continue
        m = RUN_DIR_RE.match(d.name)
        if not m:
            continue
        csv_path = d / "test_per_sample.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df["__seed__"] = int(m.group("seed"))
        by_model[m.group("model")].append(df)
    return {k: pd.concat(v, ignore_index=True) for k, v in by_model.items()}


def per_sample_seed_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Average Dice/IoU/HD95 across seeds for each (patient, view, phase,
    class_name) sample. Returns one row per sample."""
    group_cols = ["patient", "view", "phase", "class_name"]
    metrics = ["dice", "iou", "hd95"]
    metrics = [m for m in metrics if m in df.columns]
    out = (df.groupby(group_cols, as_index=False)[metrics]
             .mean(numeric_only=True))
    return out


def add_foreground_mean(df: pd.DataFrame, foreground: List[str]) -> pd.DataFrame:
    """For each (patient, view, phase), add a synthetic row class_name=='_foreground'
    holding the mean over the foreground classes."""
    fg = df[df["class_name"].isin(foreground)].copy()
    g = (fg.groupby(["patient", "view", "phase"], as_index=False)
            [["dice", "iou", "hd95"]].mean(numeric_only=True))
    g["class_name"] = "_foreground"
    return pd.concat([df, g], ignore_index=True)


def star(p: float) -> str:
    if not np.isfinite(p): return "ns"
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def panel(ax, vals_b: np.ndarray, vals_k: np.ndarray, title: str,
          ylabel: str, lower_better: bool = False) -> None:
    """One box+strip panel comparing baseline vs KAN on a single metric/class."""
    # Strip clean of NaNs (pair-wise)
    mask = np.isfinite(vals_b) & np.isfinite(vals_k)
    vb = vals_b[mask]
    vk = vals_k[mask]

    positions = [1, 2]
    box_data = [vb, vk]
    colours_local = [COLOURS["baseline_unet"], COLOURS["kan_unet"]]
    labels_local = [LABELS["baseline_unet"], LABELS["kan_unet"]]

    bp = ax.boxplot(
        box_data, positions=positions, widths=0.55,
        patch_artist=True, showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white",
                       markeredgecolor="black", markersize=5),
        medianprops=dict(color="black", linewidth=1.3),
        whiskerprops=dict(color="black", linewidth=0.9),
        capprops=dict(color="black", linewidth=0.9),
        flierprops=dict(marker="", markersize=0),  # hide outlier dots;
                                                    # we show all points via strip
    )
    for patch, c in zip(bp["boxes"], colours_local):
        patch.set_facecolor(c)
        patch.set_alpha(0.35)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)

    # Strip plot (jittered scatter)
    rng = np.random.default_rng(0)
    for pos, vals, c in zip(positions, box_data, colours_local):
        if vals.size == 0:
            continue
        jx = pos + rng.uniform(-0.13, 0.13, size=vals.size)
        ax.scatter(jx, vals, s=6, alpha=0.45, color=c,
                   edgecolors="none", zorder=3)

    # Significance bracket
    if vb.size >= 2 and vk.size >= 2:
        try:
            stat, pval = wilcoxon(vb, vk, alternative="two-sided",
                                   zero_method="zsplit")
        except ValueError:
            pval = float("nan")
        all_vals = np.concatenate([vb, vk])
        ymax = float(np.nanmax(all_vals))
        ymin = float(np.nanmin(all_vals))
        span = max(ymax - ymin, 1e-6)
        bar_y = ymax + 0.05 * span
        tick_h = 0.02 * span
        ax.plot([1, 1, 2, 2],
                [bar_y, bar_y + tick_h, bar_y + tick_h, bar_y],
                color="black", linewidth=0.9)
        ax.text(1.5, bar_y + tick_h * 1.2, star(pval) +
                (f"  p={pval:.1e}" if np.isfinite(pval) else ""),
                ha="center", va="bottom", fontsize=8.5)
        ax.set_ylim(top=bar_y + tick_h * 4)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels_local, fontsize=9)
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    arrow = "$\\downarrow$ better" if lower_better else "$\\uparrow$ better"
    ax.text(0.02, 0.98, arrow, transform=ax.transAxes,
            fontsize=8, va="top", ha="left", color="#333",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="#cccccc", alpha=0.8))


def main(args):
    runs_root = Path(args.runs_root)
    raw_by_model = load_per_sample_csvs(runs_root)
    if not raw_by_model:
        print(f"No test_per_sample.csv found under {runs_root}")
        return

    print("Loaded per-sample CSVs:")
    for m, df in raw_by_model.items():
        n_seeds = df["__seed__"].nunique()
        print(f"  {m}: {len(df):>6d} rows across {n_seeds} seeds")

    # Average over seeds per sample, add foreground-mean rows
    averaged: Dict[str, pd.DataFrame] = {}
    for model, df in raw_by_model.items():
        avg = per_sample_seed_mean(df)
        avg = add_foreground_mean(avg, foreground=["LV_endocardium",
                                                    "LV_myocardium",
                                                    "left_atrium"])
        averaged[model] = avg

    if not {"baseline_unet", "kan_unet"}.issubset(averaged.keys()):
        print("Need both baseline_unet and kan_unet runs. Got:",
              list(averaged.keys()))
        return

    df_b = averaged["baseline_unet"]
    df_k = averaged["kan_unet"]

    # Build the 2x4 figure
    classes = ["LV_endocardium", "LV_myocardium", "left_atrium", "_foreground"]
    metrics = [("dice", "Dice", False),
               ("hd95", "HD95 (px)", True)]

    fig, axes = plt.subplots(2, 4, figsize=(14, 8), dpi=150)
    for r, (metric, ylabel, lower_better) in enumerate(metrics):
        for c, cls in enumerate(classes):
            # Align by (patient, view, phase)
            sub_b = df_b[df_b["class_name"] == cls].copy()
            sub_k = df_k[df_k["class_name"] == cls].copy()
            merged = sub_b.merge(sub_k, on=["patient", "view", "phase"],
                                  suffixes=("_b", "_k"), how="inner")
            vals_b = merged[f"{metric}_b"].to_numpy()
            vals_k = merged[f"{metric}_k"].to_numpy()
            panel(axes[r, c], vals_b, vals_k,
                  title=f"{PRETTY_CLASS[cls]}", ylabel=ylabel,
                  lower_better=lower_better)

    fig.suptitle("Per-sample test-set distributions, 5-seed mean per sample",
                 fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.965])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "per_sample_distribution.pdf"
    png_path = out_dir / "per_sample_distribution.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote: {pdf_path}")
    print(f"Wrote: {png_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs_root", default="./runs")
    p.add_argument("--out_dir", default="runs/aggregate")
    args = p.parse_args()
    main(args)