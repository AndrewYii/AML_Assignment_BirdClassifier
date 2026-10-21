"""Architecture diagram of the final model, ConvNeXt-B (report figure).

Top row : the network with tensor sizes for a 384 x 384 input (verified with timm).
Bottom  : one ConvNeXt block, the repeated unit of every stage (36 blocks in total).

Usage:  .venv/Scripts/python analysis/arch_diagram.py   -> results/figures/architecture.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.utils import ROOT

CNN, NEW, GREY, BLOCK, INK = "#d6e4fb", "#fde3d6", "#eeeeee", "#f3f7fe", "#111111"


def box(ax, x, y, w, h, title, sub="", color=GREY, size=10.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="#555555", lw=0.9))
    ax.text(x + w / 2, y + h * (0.66 if sub else 0.5), title, ha="center", va="center", fontsize=size,
            fontweight="bold", color=INK)
    if sub:
        ax.text(x + w / 2, y + h * 0.29, sub, ha="center", va="center", fontsize=size - 1.2, color="#333333")


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, lw=0.9, color="#444444"))


def main():
    fig, ax = plt.subplots(figsize=(11.5, 3.6))  # drawn in 13 x 3.9 data units; a smaller canvas = larger text in print
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.9)
    ax.axis("off")

    ax.text(0.15, 3.62, "ConvNeXt-B (88M parameters, pretrained on ImageNet-22k, fine-tuned at 384 × 384)",
            fontsize=12, fontweight="bold", color="#1f3c88")
    y, h, w, gap = 2.35, 1.0, 1.42, 0.2
    stages = [("Input", "384×384×3", GREY), ("Stem", "4×4 conv, s=4\n96×96×128", CNN),
              ("Stage 1", "3 blocks\n96×96×128", CNN), ("Stage 2", "3 blocks\n48×48×256", CNN),
              ("Stage 3", "27 blocks\n24×24×512", CNN), ("Stage 4", "3 blocks\n12×12×1024", CNN),
              ("Avg-pool + LN", "global average\n1024", CNN), ("New head", "FC 1024→200\n+ softmax", NEW)]
    x = 0.15
    centers = []
    for i, (t, s, c) in enumerate(stages):
        box(ax, x, y, w, h, t, s, c)
        centers.append(x + w / 2)
        if i:
            arrow(ax, x - gap, y + h / 2, x, y + h / 2)
        x += w + gap
    ax.text(x - gap + 0.05, y + h / 2, "200\nspecies", va="center", fontsize=9.5, color="#333")
    ax.text(centers[-1], y - 0.17, "only new layer\n(randomly initialised)", ha="center", va="top", fontsize=9,
            color="#8a4b1f")
    ax.text(centers[1] + (centers[5] - centers[1]) / 2, y - 0.17,
            "pretrained backbone: fine-tuned with a 10× smaller learning rate (5e-5) than the new head (5e-4)",
            ha="center", va="top", fontsize=9.4, color="#1f3c88")
    # downsampling note between stages
    for a, b in ((2, 3), (3, 4), (4, 5)):
        ax.text((centers[a] + centers[b]) / 2, y + h + 0.06, "↓2", ha="center", fontsize=9, color="#666")

    # ---- one ConvNeXt block ----
    by, bh, bw, bgap = 0.25, 0.62, 1.5, 0.22
    ax.text(0.15, 1.12, "One ConvNeXt block (×36): residual unit with a large 7×7 kernel", fontsize=10.5, style="italic",
            color="#333")
    steps = [("7×7 depthwise", "spatial mixing"), ("LayerNorm", ""), ("1×1 conv C→4C", "expand"),
             ("GELU", ""), ("1×1 conv 4C→C", "project"), ("+ block input", "residual")]
    bx = 0.15
    for i, (t, s) in enumerate(steps):
        box(ax, bx, by, bw, bh, t, s, BLOCK, size=9.6)
        if i:
            arrow(ax, bx - bgap, by + bh / 2, bx, by + bh / 2)
        bx += bw + bgap
    ax.annotate("", xy=(centers[4], y - 0.55), xytext=(bx - bgap, by + bh + 0.05),
                arrowprops=dict(arrowstyle="-", ls=":", color="#888"))

    fig.tight_layout()
    out = ROOT / "results/figures/architecture.png"
    fig.savefig(out, dpi=220)
    print("saved", out)


if __name__ == "__main__":
    main()
