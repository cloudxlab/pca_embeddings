#!/usr/bin/env python3
"""Generate publication-quality figures for the PCA embedding paper."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

# Color-blind safe palette (Wong 2011)
BLUE   = "#0072B2"
ORANGE = "#E69F00"
GREEN  = "#009E73"
RED    = "#D55E00"
PURPLE = "#CC79A7"
GRAY   = "#999999"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

W = 3.5   # single-column width (inches)
H = 2.6   # height

# ── Data ────────────────────────────────────────────────────────────────────

DIMS = [16, 32, 48, 64, 96, 128, 192, 256]

# H1 — PCA (query+corpus fit)
map_pca_both = [0.8071, 0.9103, 0.9083, 0.9011, 0.8982, 0.8956, 0.8938, 0.8970]

# H3 — PCA (corpus-only fit)
map_pca_corp = [0.8496, 0.9203, 0.9142, 0.9137, 0.9087, 0.9063, 0.8981, 0.8943]

# Baseline
MAP_BASE = 0.8750

# H2 — Random Projection MAP
map_rp = [0.2498, 0.3582, 0.5435, 0.5128, 0.6225, 0.5767, 0.7594, 0.7622]

# SimGap
simgap_pca  = [0.6088, 0.5425, 0.4869, 0.4447, 0.3976, 0.3687, 0.3349, 0.3159]
simgap_corp = [0.7106, 0.6145, 0.5537, 0.5115, 0.4623, 0.4288, 0.3905, 0.3697]
SIMGAP_BASE = 0.2500

# H5 — topic diversity
topics      = [5, 10, 15, 20]
base_map_h5 = [0.9241, 0.8820, 0.8868, 0.8750]
pca_map_h5  = [0.9548, 0.9063, 0.9148, 0.9141]
gains_h5    = [p - b for p, b in zip(pca_map_h5, base_map_h5)]

# Variance explained
variance = [35.6, 50.8, 60.4, 67.4, 77.4, 84.1, 92.5, 97.1]


# ── Fig 1: MAP vs PCA Dimensions ────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(W, H))

ax.axhline(MAP_BASE, color=GRAY, linewidth=1.4, linestyle="--", label="Baseline (1536 dims)", zorder=1)
ax.plot(DIMS, map_pca_corp, color=BLUE,   marker="o", markersize=4, linewidth=1.6,
        label="PCA — corpus-only fit", zorder=3)
ax.plot(DIMS, map_pca_both, color=ORANGE, marker="s", markersize=4, linewidth=1.6,
        label="PCA — query+corpus fit", zorder=2)

ax.set_xlabel("PCA Dimensions")
ax.set_ylabel("Mean Average Precision (MAP)")
ax.set_title("Fig. 1 — MAP vs PCA Dimensionality")
ax.set_xticks(DIMS)
ax.set_xticklabels([str(d) for d in DIMS])
ax.set_ylim(0.76, 0.95)
ax.legend(loc="lower right", framealpha=0.9)

# Annotate best point
best_idx = map_pca_corp.index(max(map_pca_corp))
ax.annotate(f"  {max(map_pca_corp):.4f}",
            xy=(DIMS[best_idx], max(map_pca_corp)),
            fontsize=7, color=BLUE, va="center")

fig.tight_layout()
fig.savefig(OUT / "fig1_map_vs_dims.png", dpi=300, bbox_inches="tight")
plt.close()
print("Fig 1 saved.")


# ── Fig 2: PCA vs Random Projection ─────────────────────────────────────────

fig, ax = plt.subplots(figsize=(W, H))

x = np.arange(len(DIMS))
w = 0.28

ax.bar(x - w, map_pca_corp, w, color=BLUE,   label="PCA (corpus fit)", zorder=2)
ax.bar(x,     map_pca_both, w, color=ORANGE,  label="PCA (both fit)",   zorder=2)
ax.bar(x + w, map_rp,       w, color=RED,     label="Random Projection",zorder=2)
ax.axhline(MAP_BASE, color=GRAY, linewidth=1.2, linestyle="--", label="Baseline", zorder=1)

ax.set_xlabel("Reduced Dimensions")
ax.set_ylabel("MAP")
ax.set_title("Fig. 2 — PCA vs Random Projection")
ax.set_xticks(x)
ax.set_xticklabels([str(d) for d in DIMS])
ax.set_ylim(0.0, 1.0)
ax.legend(loc="lower right", framealpha=0.9, ncol=2)

fig.tight_layout()
fig.savefig(OUT / "fig2_pca_vs_rp.png", dpi=300, bbox_inches="tight")
plt.close()
print("Fig 2 saved.")


# ── Fig 3: SimGap across dimensions ─────────────────────────────────────────

fig, ax = plt.subplots(figsize=(W, H))

ax.axhline(SIMGAP_BASE, color=GRAY, linewidth=1.4, linestyle="--",
           label=f"Baseline SimGap ({SIMGAP_BASE:.2f})", zorder=1)
ax.plot(DIMS, simgap_corp, color=BLUE,   marker="o", markersize=4, linewidth=1.6,
        label="PCA — corpus-only fit", zorder=3)
ax.plot(DIMS, simgap_pca,  color=ORANGE, marker="s", markersize=4, linewidth=1.6,
        label="PCA — query+corpus fit", zorder=2)

ax.set_xlabel("PCA Dimensions")
ax.set_ylabel("Similarity Gap (Rel − Irrel)")
ax.set_title("Fig. 3 — Similarity Gap vs PCA Dimensionality")
ax.set_xticks(DIMS)
ax.set_xticklabels([str(d) for d in DIMS])
ax.set_ylim(0.1, 0.82)
ax.legend(loc="upper right", framealpha=0.9)

# Annotate 2.5× label
ax.annotate("2.5× baseline", xy=(16, simgap_corp[0]),
            xytext=(30, 0.74), fontsize=7, color=BLUE,
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8))

fig.tight_layout()
fig.savefig(OUT / "fig3_simgap.png", dpi=300, bbox_inches="tight")
plt.close()
print("Fig 3 saved.")


# ── Fig 4: PCA Gain vs Topic Diversity ──────────────────────────────────────

fig, ax = plt.subplots(figsize=(W, H))

bars = ax.bar(topics, gains_h5, width=2.5, color=[BLUE, ORANGE, GREEN, RED], zorder=2)
ax.set_xlabel("Number of Clinical Topics in Corpus")
ax.set_ylabel("MAP Gain over Baseline (PCA-32)")
ax.set_title("Fig. 4 — PCA Gain vs Corpus Diversity")
ax.set_xticks(topics)
ax.set_ylim(0, 0.055)

for bar, g in zip(bars, gains_h5):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
            f"+{g:.3f}", ha="center", va="bottom", fontsize=8)

fig.tight_layout()
fig.savefig(OUT / "fig4_topic_diversity.png", dpi=300, bbox_inches="tight")
plt.close()
print("Fig 4 saved.")


# ── Fig 5: Variance Explained + MAP (dual axis) ──────────────────────────────

fig, ax1 = plt.subplots(figsize=(W, H))
ax2 = ax1.twinx()

ln1 = ax1.plot(DIMS, variance, color=GREEN, marker="^", markersize=4,
               linewidth=1.6, label="Cumulative variance (%)")
ln2 = ax2.plot(DIMS, map_pca_corp, color=BLUE, marker="o", markersize=4,
               linewidth=1.6, linestyle="--", label="MAP (corpus-only fit)")
ax2.axhline(MAP_BASE, color=GRAY, linewidth=1.2, linestyle=":",
            label=f"Baseline MAP ({MAP_BASE:.4f})")

ax1.set_xlabel("PCA Dimensions")
ax1.set_ylabel("Cumulative Variance Explained (%)", color=GREEN)
ax2.set_ylabel("MAP", color=BLUE)
ax1.set_title("Fig. 5 — Variance Explained vs MAP Trade-off")
ax1.set_xticks(DIMS)
ax1.set_xticklabels([str(d) for d in DIMS])
ax1.tick_params(axis="y", labelcolor=GREEN)
ax2.tick_params(axis="y", labelcolor=BLUE)
ax1.set_ylim(20, 110)
ax2.set_ylim(0.76, 0.95)

lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc="lower right", framealpha=0.9, fontsize=7)

fig.tight_layout()
fig.savefig(OUT / "fig5_variance_map.png", dpi=300, bbox_inches="tight")
plt.close()
print("Fig 5 saved.")

print(f"\nAll figures saved to ./{OUT}/")
