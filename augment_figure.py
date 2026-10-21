"""Final data augmentation figure: one training photo and seven random training views.

Uses the exact transforms of the final recipe (src/data.py build_transforms, img_size 384, aug "basic").
Usage:  .venv/Scripts/python analysis/augment_figure.py   -> results/figures/augmentation_final.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import torch
from PIL import Image

from src.data import build_transforms
from src.utils import ROOT

IMAGE = "Painted_Bunting_0006_2862481106.jpg"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def to_img(x):
    """Undo the ImageNet normalisation so the tensor can be shown as a photo."""
    return (x * torch.tensor(STD)[:, None, None] + torch.tensor(MEAN)[:, None, None]).clamp(0, 1).permute(1, 2, 0)


def main():
    train_tf, _ = build_transforms(384, "basic", MEAN, STD, crop_pct=0.875)
    img = Image.open(ROOT / "dataset/Train" / IMAGE).convert("RGB")
    torch.manual_seed(3)  # fixed so the figure is the same every time
    tiles = [(img, "Original photo")] + [(to_img(train_tf(img)), f"Training view {k}") for k in range(1, 8)]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.2))
    for ax, (im, title) in zip(axes.flat, tiles):
        ax.imshow(im)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.tight_layout()
    out = ROOT / "results/figures/augmentation_final.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
