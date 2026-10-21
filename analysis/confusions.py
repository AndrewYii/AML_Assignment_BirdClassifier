"""Error analysis for one evaluated run: worst species, most-confused pairs, example grid.

Usage
    .venv/Scripts/python analysis/confusions.py final

Outputs
    results/<run>/worst_classes.csv, results/<run>/confused_pairs.csv
    results/figures/errors_<run>.png : each row = a confused pair (true example | predicted-class example)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.data import read_split
from src.utils import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--top", type=int, default=6)
    args = ap.parse_args()
    res = ROOT / "results" / args.run
    names = json.loads((ROOT / "class_names.json").read_text())

    per_class = pd.read_csv(res / "per_class.csv").sort_values(["accuracy", "f1"])
    per_class.head(10).to_csv(res / "worst_classes.csv", index=False)
    print("Worst classes:\n", per_class.head(10)[["class_name", "support", "accuracy", "f1"]].to_string(index=False))

    cm = np.load(res / "confusion.npy").astype(int)
    np.fill_diagonal(cm, 0)
    flat = np.argsort(-cm, axis=None)[: args.top * 2]
    pairs = []
    for idx in flat:
        t, p = np.unravel_index(idx, cm.shape)
        if cm[t, p] == 0:
            break
        pairs.append({"true": names[t], "predicted": names[p], "count": int(cm[t, p]), "t": t, "p": p})
    pairs = pairs[: args.top]
    pd.DataFrame(pairs).drop(columns=["t", "p"]).to_csv(res / "confused_pairs.csv", index=False)
    print("Most confused pairs:\n", pd.DataFrame(pairs)[["true", "predicted", "count"]].to_string(index=False))

    # one training image per class for the "what it was confused with" column
    train_example = {}
    for n, y in read_split(ROOT / "dataset/train.txt"):
        train_example.setdefault(y, n)
    preds = pd.read_csv(res / "preds.csv")

    # wide layout for a 4-page report: 3 pairs per row, each pair = (misclassified test image | predicted class)
    per_row = 3
    n_rows = int(np.ceil(len(pairs) / per_row))
    fig, axes = plt.subplots(n_rows, per_row * 2, figsize=(2.2 * per_row * 2, 2.3 * n_rows))
    axes = np.atleast_2d(axes)
    for a in axes.ravel():
        a.axis("off")

    def show(ax, path):
        # square centre crop so every tile has the same size
        im = Image.open(path).convert("RGB")
        s = min(im.size)
        left, top = (im.width - s) // 2, (im.height - s) // 2
        ax.imshow(im.crop((left, top, left + s, top + s)))

    for k, pr in enumerate(pairs):
        r, c = divmod(k, per_row)
        wrong = preds[(preds.target == pr["t"]) & (preds.pred == pr["p"])].iloc[0]
        show(axes[r, 2 * c], ROOT / "dataset/Test" / wrong.image)
        axes[r, 2 * c].set_title(f"true: {pr['true']}", fontsize=8, color="#008300")
        show(axes[r, 2 * c + 1], ROOT / "dataset/Train" / train_example[pr["p"]])
        axes[r, 2 * c + 1].set_title(f"predicted: {pr['predicted']} (x{pr['count']})", fontsize=8, color="#e34948")
    fig.tight_layout()
    out = ROOT / "results" / "figures" / f"errors_{args.run}.png"
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
