"""How the 5-fold cross-validation and the final ensemble work (diagram with the real per-fold numbers).

Usage:  .venv/Scripts/python analysis/cv_diagram.py   -> results/figures/cv_diagram.png
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.utils import ROOT

TRAIN, VAL, TEST, ENS, INK = "#9cc0f5", "#f29a8f", "#d9d9d9", "#9fdcc4", "#111111"


def box(ax, x, y, w, h, color, text="", size=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.005,rounding_size=0.06", fc=color, ec="#555", lw=0.8))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK,
                fontweight="bold" if bold else "normal")


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10, lw=1, color="#444"))


def main():
    cv = json.loads((ROOT / "results/cv5_summary.json").read_text())["final"]
    val = {r["fold"]: r["top1"] for r in cv["per_fold"]}
    y = np.array([int(l.split()[1]) for l in (ROOT / "dataset/test.txt").read_text().splitlines()])
    test = {i: float((np.load(ROOT / "results" / f"cv5_final_f{i}" / "logits.npy").argmax(1) == y).mean() * 100)
            for i in range(1, 6)}
    ens = json.loads((ROOT / "results/final/metrics.json").read_text())["top1"] * 100

    fig, ax = plt.subplots(figsize=(11, 3.9))
    ax.set_xlim(0, 11), ax.set_ylim(0, 3.9), ax.axis("off")
    ax.text(0.05, 3.72, "Training set: 4,829 images, split into 5 stratified folds (~966 each)", fontsize=10, fontweight="bold")
    fw, fh, x0 = 0.78, 0.42, 0.75
    for r in range(5):  # run r+1 validates on fold r+1
        yy = 3.08 - r * 0.55
        ax.text(0.05, yy + fh / 2, f"Run {r + 1}", va="center", fontsize=9.5)
        for f in range(5):
            is_val = f == r
            box(ax, x0 + f * (fw + 0.04), yy, fw, fh, VAL if is_val else TRAIN, "validate" if is_val else "train", 8.5,
                bold=is_val)
        arrow(ax, (x0 + 5 * (fw + 0.04) + 0.02, yy + fh / 2), (5.35, yy + fh / 2))
        box(ax, 5.4, yy, 1.55, fh, "#ffffff", f"model {r + 1}: val {val[r + 1]:.1f}%", 8.5)
        arrow(ax, (6.97, yy + fh / 2), (7.55, 1.98))
    box(ax, 7.6, 1.62, 1.55, 0.72, ENS, "average the\n5 models'\nprobabilities", 8.5, bold=True)
    arrow(ax, (9.17, 1.98), (9.45, 1.98))
    box(ax, 9.5, 1.5, 1.45, 0.96, ENS, f"final model\ntest {ens:.2f}%", 9.5, bold=True)
    box(ax, 7.6, 0.2, 3.35, 0.5, TEST, "Test set: 1,204 images, never used during CV", 8.5)
    ax.text(0.75, 0.42, f"Mean validation accuracy of the 5 runs: {cv['val_top1_mean']:.2f} ± {cv['val_top1_std']:.2f}%   ·   "
                        f"each model alone on test: {min(test.values()):.1f}–{max(test.values()):.1f}%",
            fontsize=9, va="center", color="#333")
    arrow(ax, (9.27, 0.72), (9.9, 1.48))
    fig.tight_layout()
    out = ROOT / "results/figures/cv_diagram.png"
    fig.savefig(out, dpi=200)
    print("saved", out)


if __name__ == "__main__":
    main()
