"""E9 figure: train/test accuracy vs training-set size (25 / 50 / 100 % of every class).

Usage
    .venv/Scripts/python analysis/data_size.py
The 100 % point is the mean +/- std of the three batch-16 seeds (E5).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import ROOT


def top1(r):
    return json.loads((ROOT / "results" / r / "metrics.json").read_text())["top1"] * 100


def train_acc(r):
    return pd.read_csv(ROOT / "results" / r / "history.csv").train_acc.iloc[-1] * 100


def main():
    seeds = ["e5_batch16", "e5_batch16_seed1", "e5_batch16_seed2"]
    full_te = [top1(r) for r in seeds]
    rows = [(25, top1("e9_data_25"), train_acc("e9_data_25"), 0.0),
            (50, top1("e9_data_50"), train_acc("e9_data_50"), 0.0),
            (100, float(np.mean(full_te)), float(np.mean([train_acc(r) for r in seeds])), float(np.std(full_te, ddof=1)))]
    x = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(x, [r[2] for r in rows], "o-", color="#2a78d6", lw=2, label="train accuracy")
    ax.plot(x, [r[1] for r in rows], "o-", color="#eb6834", lw=2, label="test top-1")
    ax.errorbar(100, rows[-1][1], yerr=rows[-1][3], color="#eb6834", capsize=4)
    for i, (xi, te, tr, _) in enumerate(rows):
        ax.annotate(f"{te:.1f}%", (xi, te), textcoords="offset points", xytext=(0, -15), ha="center", fontsize=9)
        last = i == len(rows) - 1
        ax.annotate(f"gap {tr - te:.1f}", (xi, (tr + te) / 2), textcoords="offset points",
                    xytext=(-8 if last else 8, 0), ha="right" if last else "left", fontsize=8, color="#52514e")
    ax.set_xticks(x, ["25%\n(~6 img/class)", "50%\n(~12 img/class)", "100%\n(~24 img/class)"])
    ax.set_xlim(15, 110)
    ax.set_ylim(60, 102)
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Training-set size (ConvNeXt-T, 224 px, same recipe)", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color="#e4e3df")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(ROOT / "results/figures/e9_data_size.png", dpi=200)
    print("saved results/figures/e9_data_size.png")


if __name__ == "__main__":
    main()
