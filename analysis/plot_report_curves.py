"""Report-style learning curves: one panel per run, large round markers (one per epoch),
blue = training, red = test. Easy to read at small size in a 4-page report.

Usage
    .venv/Scripts/python analysis/plot_report_curves.py e2_baseline_resnet50 e15b_ema_convnext_base \
        --labels "Baseline ResNet-50" "Final CNN ConvNeXt-B" --name baseline_vs_final --metric loss
    --metric loss | acc | both | row   (both = loss on top, accuracy below; row = all panels in one compact row,
                                        sized for a full-width figure in the 4-page report)
Writes results/figures/report_<name>_<metric>.png
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd

from src.utils import ROOT

TRAIN, TEST = "#1f3cff", "#e8221b"  # strong blue / red, as in common report figures
STYLE = dict(marker="o", markersize=6, linewidth=1.6, alpha=0.85)


def panel(ax, h, metric, title, small=False):
    style = dict(STYLE, markersize=3.2, linewidth=1.3) if small else STYLE
    if metric == "loss":
        ax.plot(h.epoch, h.train_loss, color=TRAIN, label="Training loss", **style)
        ax.plot(h.epoch, h.test_loss, color=TEST, label="Test loss", **style)
        ax.set_ylabel("Loss")
        ax.set_ylim(bottom=0)
    else:
        ax.plot(h.epoch, h.train_acc * 100, color=TRAIN, label="Training accuracy", **style)
        ax.plot(h.epoch, h.test_top1 * 100, color=TEST, label="Test top-1 accuracy", **style)
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 102)
    if "phase" in h and (h.phase == "linear_probe").any() and (h.phase != "linear_probe").any():
        k = int((h.phase == "linear_probe").sum())
        ax.axvline(k + 0.5, color="#555", ls="--", lw=1)
        ax.text(k + 0.8, ax.get_ylim()[1] * 0.92, "backbone unfrozen", fontsize=9, color="#555")
    ax.set_title(title, fontsize=10 if small else 13)
    ax.set_xlabel("Epoch")
    ax.grid(True, color="#d9d9d9", linewidth=0.8)
    ax.legend(loc="upper right" if metric == "loss" else "lower right", fontsize=8.5 if small else 10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--name", required=True)
    ap.add_argument("--metric", default="loss", choices=["loss", "acc", "both", "row"])
    args = ap.parse_args()
    labels = args.labels or args.runs
    hists = [pd.read_csv(ROOT / "results" / r / "history.csv") for r in args.runs]
    metrics = ["loss", "acc"] if args.metric in ("both", "row") else [args.metric]
    if args.metric == "row":
        with plt.rc_context({"font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5}):
            fig, axes = plt.subplots(1, len(hists) * 2, figsize=(10.5, 2.75))
            for j, (h, lab) in enumerate(zip(hists, labels)):
                for i, met in enumerate(metrics):
                    panel(axes[2 * j + i], h, met, f"{lab}: {'loss' if met == 'loss' else 'accuracy'}", small=True)
            fig.tight_layout(w_pad=0.6)
        out = ROOT / "results" / "figures" / f"report_{args.name}_row.png"
        fig.savefig(out, dpi=220)
        print("saved", out)
        return

    fig, axes = plt.subplots(len(metrics), len(hists), figsize=(6.2 * len(hists), 4.3 * len(metrics)),
                             squeeze=False)
    for j, (h, lab) in enumerate(zip(hists, labels)):
        for i, met in enumerate(metrics):
            panel(axes[i, j], h, met, f"{lab}: {'loss' if met == 'loss' else 'accuracy'}")
    fig.tight_layout()
    out = ROOT / "results" / "figures" / f"report_{args.name}_{args.metric}.png"
    fig.savefig(out, dpi=200)
    print("saved", out)


if __name__ == "__main__":
    main()
