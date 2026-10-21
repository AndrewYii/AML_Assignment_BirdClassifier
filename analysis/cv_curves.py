"""Learning curves of the 5-fold cross-validation (final ConvNeXt-B recipe).

For each epoch: training loss/accuracy and validation loss/accuracy, as the mean over the 5 folds with a band
for the spread (min-max). The test set is never used during cross-validation, so it has no per-epoch curve here:
each fold model is scored on the test set once, after its last epoch (dots), and the average of the five
models (the final ensemble) is the star.

Usage:  .venv/Scripts/python analysis/cv_curves.py   -> results/figures/cv5_curves.png
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import ROOT

TRAIN, VAL, TEST = "#1f3cff", "#e8221b", "#1baf7a"


def band(ax, x, runs, color, label):
    m = runs.mean(0)
    ax.fill_between(x, runs.min(0), runs.max(0), color=color, alpha=0.15, lw=0)
    ax.plot(x, m, color=color, marker="o", markersize=3.2, lw=1.4, label=label)


def main():
    hs = [pd.read_csv(ROOT / "results" / f"cv5_final_f{i}" / "history.csv") for i in range(1, 6)]
    x = hs[0].epoch.values
    get = lambda col, k=1.0: np.array([h[col].values * k for h in hs])
    y = np.array([int(l.split()[1]) for l in (ROOT / "dataset/test.txt").read_text().splitlines()])
    test = [float((np.load(ROOT / "results" / f"cv5_final_f{i}" / "logits.npy").argmax(1) == y).mean() * 100)
            for i in range(1, 6)]
    ens = json.loads((ROOT / "results/final/metrics.json").read_text())["top1"] * 100

    with plt.rc_context({"font.size": 9}):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.6))
        band(a1, x, get("train_loss"), TRAIN, "Training loss (4 folds)")
        band(a1, x, get("test_loss"), VAL, "Validation loss (held-out fold)")
        a1.set_xlabel("Epoch"), a1.set_ylabel("Loss"), a1.set_ylim(bottom=0)
        a1.set_title("ConvNeXt-B, 5-fold CV: loss (mean, band = min–max over folds)", fontsize=9.5)
        a1.legend(fontsize=8.5)

        band(a2, x, get("train_acc", 100), TRAIN, "Training accuracy")
        band(a2, x, get("test_top1", 100), VAL, "Validation accuracy (held-out fold)")
        a2.scatter([x[-1] + 0.6] * 5, test, color=TEST, s=18, zorder=3, label="Test accuracy of each fold model (once)")
        a2.scatter([x[-1] + 1.4], [ens], color=TEST, marker="*", s=140, zorder=3,
                   label=f"Test accuracy, 5 models averaged ({ens:.2f}%)")
        a2.set_xlabel("Epoch"), a2.set_ylabel("Accuracy (%)"), a2.set_ylim(0, 102), a2.set_xlim(0, x[-1] + 2.5)
        a2.set_title("ConvNeXt-B, 5-fold CV: accuracy", fontsize=9.5)
        a2.legend(fontsize=8, loc="lower right")
        for a in (a1, a2):
            a.grid(True, color="#d9d9d9", lw=0.8)
        # zoomed inset: the flat part where validation and test meet
        ins = a2.inset_axes([0.42, 0.42, 0.33, 0.28])
        band(ins, x, get("test_top1", 100), VAL, None)
        ins.scatter([x[-1] + 0.6] * 5, test, color=TEST, s=10, zorder=3)
        ins.scatter([x[-1] + 1.4], [ens], color=TEST, marker="*", s=60, zorder=3)
        ins.set_xlim(4, x[-1] + 2.5), ins.set_ylim(91.5, 96), ins.tick_params(labelsize=7)
        ins.set_title("epochs 5–30, zoomed", fontsize=7)
        fig.tight_layout()
    out = ROOT / "results/figures/cv5_curves.png"
    fig.savefig(out, dpi=200)
    print("saved", out, "| val mean last", round(get("test_top1", 100)[:, -1].mean(), 2), "| fold test", [round(t, 2) for t in test])


if __name__ == "__main__":
    main()
