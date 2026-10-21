"""Confusion matrix of the final model with the 200 species sorted by family (report appendix).

Species are grouped by the last word of their name (Warbler, Sparrow, Gull, ...), so confusions between
look-alike species of one family appear as small off-diagonal marks inside the family's block.

Usage:  .venv/Scripts/python analysis/confusion_figure.py   -> results/figures/confusion_by_family.png
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from src.utils import ROOT


def main():
    names = json.loads((ROOT / "class_names.json").read_text())
    cm = np.load(ROOT / "results/final/confusion.npy").astype(float)
    cm = cm / cm.sum(1, keepdims=True)                      # rows = true species, normalised
    fam = [n.split()[-1] for n in names]
    order = sorted(range(len(names)), key=lambda i: (fam[i], names[i]))
    cm = cm[np.ix_(order, order)]
    fam_sorted = [fam[i] for i in order]

    off = cm.copy()
    np.fill_diagonal(off, 0)
    in_block = sum(off[i, j] for i in range(len(order)) for j in range(len(order)) if fam_sorted[i] == fam_sorted[j])

    fig, ax = plt.subplots(figsize=(7.4, 7.6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    # outline the families with at least 4 species
    start = 0
    for k in range(1, len(order) + 1):
        if k == len(order) or fam_sorted[k] != fam_sorted[start]:
            if k - start >= 4:
                ax.add_patch(plt.Rectangle((start - 0.5, start - 0.5), k - start, k - start, fill=False,
                                           ec="#e8221b", lw=0.8))
                ax.text(k + 1, (start + k) / 2, f"{fam_sorted[start]}s ({k - start})", fontsize=6.5, va="center",
                        color="#b00020")
            start = k
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("predicted species (sorted by family)")
    ax.set_ylabel("true species (sorted by family)")
    ax.set_title(f"Final model: normalised confusion matrix; {in_block / off.sum() * 100:.0f}% of the off-diagonal "
                 "mass lies inside a family block", fontsize=9)
    fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.035, pad=0.06,
                 label="fraction of the true species' test images")
    fig.tight_layout()
    out = ROOT / "results/figures/confusion_by_family.png"
    fig.savefig(out, dpi=200)
    print("saved", out, f"(in-family share of errors {in_block / off.sum() * 100:.1f}%)")


if __name__ == "__main__":
    main()
