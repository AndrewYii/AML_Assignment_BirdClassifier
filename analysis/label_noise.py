"""The final model's most confident test errors, shown with their dataset label and the model's answer.

On inspection these photos appear to show the species the model predicted (probable label noise).
Usage:  .venv/Scripts/python analysis/label_noise.py --n 4   -> results/figures/label_noise.png
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from src.utils import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()
    names = json.loads((ROOT / "class_names.json").read_text())
    p = pd.read_csv(ROOT / "results/final/preds.csv")
    worst = p[p.pred != p.target].sort_values("confidence", ascending=False).head(args.n)
    fig, axes = plt.subplots(1, args.n, figsize=(2.9 * args.n, 3.3))
    for a, r in zip(axes, worst.itertuples()):
        a.imshow(Image.open(ROOT / "dataset/Test" / r.image).convert("RGB"))
        a.set_title(f"label: {names[r.target]}\nmodel: {names[r.pred]} ({r.confidence:.0%})", fontsize=10)
        a.axis("off")
    fig.suptitle("Most confident errors of the final model: the photos appear to show the predicted species", fontsize=11)
    fig.tight_layout()
    out = ROOT / "results/figures/label_noise.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
