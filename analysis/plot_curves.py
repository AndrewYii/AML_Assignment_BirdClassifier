"""Learning curves for one or more runs, as easy-to-read small multiples.

Usage
    .venv/Scripts/python analysis/plot_curves.py e2_baseline_resnet50 e3_convnext_tiny --name e3_backbones \
        --labels ResNet-50 ConvNeXt-T

Writes two figures to results/figures/:
  curves_<name>.png       one column per model; top row = loss, bottom row = accuracy.
                          Train is blue, test is orange; the final train-test gap is annotated.
  compare_<name>.png      test top-1 of all models on one chart (only when >1 run).
A widening gap between train and test is the overfitting signal discussed in the report.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd

from src.utils import ROOT

# Bright, colour-blind-validated categorical palette (fixed order, never cycled)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TRAIN, TEST = PALETTE[0], PALETTE[1]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "lines.linewidth": 2,
    "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False,
})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--name", required=True, help="output file stem")
    ap.add_argument("--labels", nargs="*", help="panel titles (default: exp ids)")
    args = ap.parse_args()
    labels = args.labels or args.runs
    hists = [pd.read_csv(ROOT / "results" / r / "history.csv") for r in args.runs]
    out_dir = ROOT / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- small multiples: one column per model ----------
    n = len(args.runs)
    fig, axes = plt.subplots(2, n, figsize=(3.6 * n, 6.0), squeeze=False, sharey="row")
    loss_max = max(max(h.train_loss.max(), h.test_loss.max()) for h in hists)
    for j, (h, lab) in enumerate(zip(hists, labels)):
        al, aa = axes[0, j], axes[1, j]
        mk = dict(marker="o", markersize=3.2, markeredgewidth=0)  # one dot per epoch
        al.plot(h.epoch, h.train_loss, color=TRAIN, label="train", **mk)
        al.plot(h.epoch, h.test_loss, color=TEST, label="test", **mk)
        al.set_title(lab, fontsize=12, fontweight="bold", color=INK)
        al.set_ylim(0, loss_max * 1.05)
        aa.plot(h.epoch, h.train_acc * 100, color=TRAIN, label="train", **mk)
        aa.plot(h.epoch, h.test_top1 * 100, color=TEST, label="test top-1", **mk)
        if "phase" in h and (h.phase == "linear_probe").any() and (h.phase != "linear_probe").any():
            k = int((h.phase == "linear_probe").sum())  # mark where the backbone was unfrozen
            for a in (al, aa):
                a.axvline(k + 0.5, color="#52514e", ls="--", lw=1)
            aa.text(k + 0.8, 8, "backbone unfrozen", fontsize=8, color="#52514e")
        aa.set_ylim(0, 102)
        aa.set_xlabel("epoch")
        # annotate the final numbers and the overfitting gap (text in ink, not series colour)
        tr, te = h.train_acc.iloc[-1] * 100, h.test_top1.iloc[-1] * 100
        aa.text(0.97, 0.05, f"train {tr:.1f}%\ntest {te:.1f}%\ngap {tr - te:.1f} pts",
                transform=aa.transAxes, ha="right", va="bottom", fontsize=9, color=INK,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID))
    axes[0, 0].set_ylabel("loss")
    axes[1, 0].set_ylabel("accuracy (%)")
    axes[0, 0].legend(loc="upper right")
    axes[1, 0].legend(loc="center right")
    fig.tight_layout()
    out = out_dir / f"curves_{args.name}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("saved", out)

    # ---------- one chart comparing test top-1 across models ----------
    if n > 1:
        fig, ax = plt.subplots(figsize=(7, 4))
        ends = []
        for i, (h, lab) in enumerate(zip(hists, labels)):
            c = PALETTE[i % len(PALETTE)]
            ax.plot(h.epoch, h.test_top1 * 100, color=c, label=lab, marker="o", markersize=3.2, markeredgewidth=0)
            ax.scatter(h.epoch.iloc[-1], h.test_top1.iloc[-1] * 100, color=c, s=40, zorder=3,
                       edgecolor="white", linewidth=2)
            ends.append([h.test_top1.iloc[-1] * 100, h.epoch.iloc[-1]])
        lo = min(h.test_top1.iloc[len(h) // 3:].min() for h in hists) * 100
        hi = max(h.test_top1.max() for h in hists) * 100
        ax.set_ylim(max(0, lo - 5), hi + 2)
        # direct end labels (<= 4 series), nudged apart so close values never overlap
        if n <= 4:
            min_gap = (hi + 2 - max(0, lo - 5)) * 0.06
            order = sorted(range(n), key=lambda k: ends[k][0])
            label_y = [e[0] for e in ends]
            for a, b in zip(order, order[1:]):
                label_y[b] = max(label_y[b], label_y[a] + min_gap)
            for k in range(n):
                ax.annotate(f"{ends[k][0]:.1f}%", (ends[k][1], ends[k][0]), xytext=(ends[k][1] + 0.6, label_y[k]),
                            textcoords="data", va="center", fontsize=9, color=INK)
        ax.set_xlim(None, hists[0].epoch.max() + 3)
        ax.set(xlabel="epoch", ylabel="test top-1 (%)", title="Test accuracy per epoch")
        ax.legend(loc="lower right")
        fig.tight_layout()
        out = out_dir / f"compare_{args.name}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print("saved", out)


if __name__ == "__main__":
    main()
