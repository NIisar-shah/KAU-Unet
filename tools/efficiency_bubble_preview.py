"""Measured accuracy-efficiency bubble chart.

Dice and parameter counts come from the project results. Throughput is derived
from the same-GPU mean latency values reported in the manuscript efficiency
table: FPS = 1000 / latency_ms.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = [
    # registry, display label, params (M), measured latency (ms), colour, marker
    ("baseline_unet", "U-Net", 1.927, 2.53, "#8C8C8C", "o"),
    ("kan_unet", "KAN-UNet", 3.241, 4.56, "#4C78A8", "D"),
    ("kan_unet_attn", "KAU-Net", 3.274, 4.83, "#D9534F", "*"),
]

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.linewidth": 0.8,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})

aggregate = pd.read_csv(ROOT / "runs" / "aggregate" / "aggregate.csv")
overall = aggregate[aggregate["group"] == "overall"].set_index("model")

fig, ax = plt.subplots(figsize=(5.0, 3.6), constrained_layout=True)
legend_handles = []
legend_labels = []
for index, (name, label, params_m, latency_ms, colour, marker) in enumerate(MODELS):
    fps = 1000.0 / latency_ms
    dice_pct = 100 * float(overall.loc[name, "dice_mean_seed_mean"])
    handle = ax.scatter(
        fps, dice_pct,
        s=90 * params_m,
        marker=marker,
        color=colour,
        edgecolor="#222222",
        linewidth=0.8,
        alpha=0.92,
        zorder=3,
    )
    legend_handles.append(Line2D(
        [0], [0], marker=marker, linestyle="none", markersize=6.5,
        markerfacecolor=colour, markeredgecolor="#222222",
        markeredgewidth=0.7, alpha=0.92,
    ))
    legend_labels.append(label)
    offsets = {
        "baseline_unet": (-8, 10),
        "kan_unet": (10, -15),
        "kan_unet_attn": (10, 10),
    }
    dx, dy = offsets[name]
    ax.annotate(
        f"{params_m:.2f} M; {fps:.1f} FPS",
        xy=(fps, dice_pct),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="right" if dx < 0 else "left",
        va="bottom" if dy >= 0 else "top",
        fontsize=6.0,
        color=colour,
        fontweight="semibold",
        zorder=4,
    )

ax.set_xlabel("Throughput (frames per second)")
ax.set_ylabel("Test Dice (%)")
ax.set_xlim(170, 425)
ax.set_ylim(87.85, 88.62)
ax.grid(False)
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color("#303030")
    spine.set_linewidth(0.8)
legend = ax.legend(
    legend_handles,
    legend_labels,
    loc="upper right",
    fontsize=5.7,
    labelspacing=0.7,
    handletextpad=0.7,
    borderpad=0.7,
    frameon=True,
    fancybox=False,
    facecolor="white",
    edgecolor="#303030",
    framealpha=0.94,
)
legend.get_frame().set_linewidth(0.6)
fig.suptitle(
    "Accuracy–efficiency comparison across selected architectures",
    fontsize=8.2,
    fontweight="semibold",
)
base = OUT / "efficiency_bubble_measured"
fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
print(f"Saved measured efficiency chart to {base}.pdf/.svg/.png/.tiff")
