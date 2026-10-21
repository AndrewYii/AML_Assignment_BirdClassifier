"""Headline figures for the report.

1. progression.png   : test top-1 after each key decision (baseline -> final), one bar per step,
                       with the +/- 1 std seed-noise band drawn on the 3-seed reference step.
2. cnn_vs_vit.png    : CNN vs transformer under (a) the same ImageNet-1k recipe and
                       (b) strong pretraining at ~384 px with a suitable LR.

Usage
    .venv/Scripts/python analysis/story_chart.py
Steps whose results do not exist yet are skipped, so it can be re-run as experiments finish.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from src.utils import ROOT

BLUE, ORANGE, AQUA, MUTED, INK, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#9a9893", "#0b0b0b", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})


def top1(run):
    f = ROOT / "results" / run / "metrics.json"
    return json.loads(f.read_text())["top1"] * 100 if f.exists() else None


def seeds(runs):
    v = [top1(r) for r in runs]
    v = [x for x in v if x is not None]
    return (float(np.mean(v)), float(np.std(v, ddof=1))) if len(v) > 1 else (v[0], 0.0) if v else (None, 0)


def progression():
    mean16, std16 = seeds(["e5_batch16", "e5_batch16_seed1", "e5_batch16_seed2"])
    steps = [
        ("Baseline\nResNet-50, 224px", top1("e2_baseline_resnet50"), 0),
        ("Backbone\nConvNeXt-T", top1("e3_convnext_tiny"), 0),
        ("Batch 16\n(3-seed mean)", mean16, std16),
        ("Resolution\n384 px", top1("e10_res384"), 0),
        ("ConvNeXt-B\nIN-22k + EMA", top1("e15b_ema_convnext_base"), 0),
        ("Final: 5-fold\nensemble", top1("final"), 0),
    ]
    steps = [s for s in steps if s[1] is not None]

    fig, ax = plt.subplots(figsize=(7.8, 3.4))
    x = np.arange(len(steps))
    vals = [s[1] for s in steps]
    colors = [MUTED] + [BLUE] * (len(steps) - 2) + [ORANGE]
    ax.bar(x, vals, color=colors, width=0.62)
    for i, (_, v, sd) in enumerate(steps):
        if sd:
            ax.errorbar(i, v, yerr=sd, color=INK, capsize=4, lw=1.2)
        ax.text(i, v + (sd or 0) + 0.25, f"{v:.2f}", ha="center", va="bottom", fontsize=9, color=INK)
        if i:
            d = v - vals[i - 1]
            ax.text(i, 80.4, f"{d:+.2f}", ha="center", fontsize=9, color="white", fontweight="bold")
    ax.set_xticks(x, [s[0] for s in steps], fontsize=8.5)
    ax.set_ylim(80, 97)
    ax.set_ylabel("test top-1 (%)")
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(ROOT / "results/figures/progression.png", dpi=200)
    plt.close(fig)


def cnn_vs_vit():
    groups = [
        ("ImageNet-1k, 224 px, same recipe",
         [("ResNet-50", "e2_baseline_resnet50", "cnn"), ("ConvNeXt-T", "e3_convnext_tiny", "cnn"),
          ("Swin-T", "e3_swin_tiny", "vit"), ("DeiT-S", "e3_deit_small", "vit")]),
        ("ImageNet-22k, 384 px, same recipe",
         [("ConvNeXt-B", "e15b_ema_convnext_base", "cnn"), ("Swin-B", "e18_swin_base_22k_384_ema", "vit")]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True, gridspec_kw={"width_ratios": [2, 1.15]})
    for ax, (title, items) in zip(axes, groups):
        items = [(n, top1(r), k) for n, r, k in items if top1(r) is not None]
        x = np.arange(len(items))
        ax.bar(x, [v for _, v, _ in items], color=[BLUE if k == "cnn" else AQUA for _, _, k in items], width=0.6)
        for i, (_, v, _) in enumerate(items):
            ax.text(i, v + 0.2, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(x, [n for n, _, _ in items], fontsize=8)
        ax.set_xlim(-0.6, len(items) - 0.4)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(80, 97)
        ax.grid(axis="y", color=GRID)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("test top-1 (%)")
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color=BLUE, label="CNN"), Patch(color=AQUA, label="Transformer")],
                   loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "results/figures/cnn_vs_vit.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    (ROOT / "results/figures").mkdir(parents=True, exist_ok=True)
    progression()
    cnn_vs_vit()
    print("saved results/figures/progression.png, cnn_vs_vit.png")
