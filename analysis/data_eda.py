"""E0 - dataset analysis: class balance, image sizes, and look-alike species.

Outputs (results/figures/ and results/eda_stats.json):
  eda_class_counts.png   training images per class (sorted) -> imbalance
  eda_image_sizes.png    width/height scatter -> why we resize + crop
  eda_lookalikes.png     example grid of visually similar species (fine-grained challenge)
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.data import class_names_from_train, read_split
from src.utils import ROOT, save_json

FIG = ROOT / "results" / "figures"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    train = read_split(ROOT / "dataset/train.txt")
    test = read_split(ROOT / "dataset/test.txt")
    names = class_names_from_train(ROOT / "dataset/train.txt")

    tr_counts = Counter(y for _, y in train)
    te_counts = Counter(y for _, y in test)
    tr = np.array([tr_counts[c] for c in range(len(names))])
    te = np.array([te_counts[c] for c in range(len(names))])

    # ---- class balance ----
    order = np.argsort(tr)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.bar(range(len(tr)), tr[order], color="#4C72B0", width=1.0, label="train")
    ax.bar(range(len(te)), te[order], color="#DD8452", width=1.0, label="test")
    ax.axhline(tr.mean(), color="k", ls="--", lw=1, label=f"train mean = {tr.mean():.1f}")
    ax.set_xlabel("class (sorted by training count)")
    ax.set_ylabel("images")
    ax.set_title("Images per class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "eda_class_counts.png", dpi=200)
    plt.close(fig)

    # ---- image sizes ----
    sizes = []
    for name, _ in train:
        with Image.open(ROOT / "dataset/Train" / name) as im:
            sizes.append(im.size)
    sizes = np.array(sizes)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.scatter(sizes[:, 0], sizes[:, 1], s=3, alpha=0.3)
    ax.set_xlabel("width (px)")
    ax.set_ylabel("height (px)")
    ax.set_title("Training image sizes")
    fig.tight_layout()
    fig.savefig(FIG / "eda_image_sizes.png", dpi=200)
    plt.close(fig)

    # ---- look-alike species: all flycatchers, sparrows, etc. share a suffix ----
    groups = ["Flycatcher", "Sparrow", "Warbler", "Vireo"]
    picks = []
    for g in groups:
        ids = [i for i, n in enumerate(names) if n.endswith(g)][:4]
        picks.append(ids)
    first = {}
    for name, y in train:
        first.setdefault(y, name)
    fig, axes = plt.subplots(len(groups), 4, figsize=(9, 2.4 * len(groups)))
    for r, ids in enumerate(picks):
        for c in range(4):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(ids):
                ax.imshow(Image.open(ROOT / "dataset/Train" / first[ids[c]]).convert("RGB"))
                ax.set_title(names[ids[c]], fontsize=8)
    fig.suptitle("Fine-grained look-alikes: species within the same family", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "eda_lookalikes.png", dpi=150)
    plt.close(fig)

    family_counts = {g: sum(n.endswith(g) for n in names) for g in groups}
    stats = {
        "n_classes": len(names), "n_train": len(train), "n_test": len(test),
        "train_per_class": {"min": int(tr.min()), "max": int(tr.max()), "mean": float(tr.mean()),
                            "median": float(np.median(tr))},
        "test_per_class": {"min": int(te.min()), "max": int(te.max()), "mean": float(te.mean())},
        "imbalance_ratio": float(tr.max() / tr.min()),
        "width": {"min": int(sizes[:, 0].min()), "max": int(sizes[:, 0].max()), "median": float(np.median(sizes[:, 0]))},
        "height": {"min": int(sizes[:, 1].min()), "max": int(sizes[:, 1].max()), "median": float(np.median(sizes[:, 1]))},
        "classes_per_family": family_counts,
        "fewest_train": [names[i] for i in order[:5]],
    }
    save_json(stats, ROOT / "results" / "eda_stats.json")
    save_json(names, ROOT / "class_names.json")
    print(stats)


if __name__ == "__main__":
    main()
