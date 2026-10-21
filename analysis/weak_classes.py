"""Diagnose the species with the lowest per-class accuracy in the final model.

Usage
    .venv/Scripts/python analysis/weak_classes.py --k 8

For each of the k weakest species (final ensemble, results/final):
  1. test images (support)            - small support = noisy accuracy
  2. training images                  - little training data?
  3. clean training accuracy (CNN)    - train high / test low = overfitting on that species;
                                        both low = species not separable
  4. most-confused species            - a consistent partner = genuine look-alike
  5. true species in top-5, confidence of the wrong answers
  6. baseline (ResNet-50) accuracy on that species
  7. a picture of every misclassified test image (results/figures/weak_classes_errors.png)
Writes results/final/weak_classes.csv
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.data import CUBDataset, build_transforms, read_split
from src.engine import predict
from src.models import build_model, norm_stats
from src.utils import ROOT, load_config

CNN_RUN, BASELINE_RUN = "e15b_ema_convnext_base", "e2_baseline_resnet50"


def clean_train_acc(run, items, device):
    """Accuracy of one trained model on training images, with the TEST transform (no augmentation)."""
    cfg = load_config(ROOT / "runs" / run / "config.yaml")
    cfg["model"]["pretrained"] = False
    cfg["model"]["grad_ckpt"] = False
    model = build_model(cfg, 200).to(device).eval()
    model.load_state_dict(torch.load(ROOT / "runs" / run / "last.pt", map_location=device)["model"])
    _, tf = build_transforms(cfg["data"]["img_size"], "none", *norm_stats(model), cfg["data"].get("crop_pct", 0.875))
    loader = DataLoader(CUBDataset(items, ROOT / "dataset/Train", tf), batch_size=16, num_workers=0)
    logits, targets = predict(model, loader, device, desc="clean train")
    return pd.Series(logits.argmax(1) == targets).groupby(targets).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()
    names = json.loads((ROOT / "class_names.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preds = pd.read_csv(ROOT / "results/final/preds.csv")
    per = pd.read_csv(ROOT / "results/final/per_class.csv").sort_values(["accuracy", "f1"])
    weak = per.head(args.k).class_id.tolist()
    base = pd.read_csv(ROOT / f"results/{BASELINE_RUN}/per_class.csv").set_index("class_id").accuracy
    train_items = read_split(ROOT / "dataset/train.txt")
    n_train = Counter(y for _, y in train_items)
    train_acc = clean_train_acc(CNN_RUN, [(f, y) for f, y in train_items if y in weak], device)

    rows = []
    for c in weak:
        p = preds[preds.target == c]
        wrong = p[p.pred != c]
        conf_with = Counter(names[x] for x in wrong.pred).most_common(2)
        in_top5 = sum(str(c) in t.split() for t in wrong.top5)
        rows.append({
            "species": names[c], "test_imgs": len(p), "train_imgs": n_train[c],
            "test_acc": round(p.pred.eq(c).mean() * 100, 1),
            "clean_train_acc": round(train_acc.get(c, float("nan")) * 100, 1),
            "baseline_test_acc": round(base[c] * 100, 1),
            "confused_with": "; ".join(f"{n} x{k}" for n, k in conf_with),
            "true_in_top5": f"{in_top5}/{len(wrong)}",
            "wrong_conf_mean": round(wrong.confidence.mean() * 100, 0) if len(wrong) else None,
        })
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/final/weak_classes.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nall species: median training images {pd.Series(n_train).median():.0f}")

    # 7. every misclassified test image of the weak species, with what the model said
    wrong = preds[preds.target.isin(weak) & (preds.pred != preds.target)]
    n = len(wrong)
    cols = 6
    rows_ = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_, cols, figsize=(2.3 * cols, 2.55 * rows_), squeeze=False)
    for a in axes.ravel():
        a.axis("off")
    for a, r in zip(axes.ravel(), wrong.itertuples()):
        a.imshow(Image.open(ROOT / "dataset/Test" / r.image).convert("RGB"))
        a.set_title(f"true: {names[r.target]}\npred: {names[r.pred]} ({r.confidence:.0%})", fontsize=7,
                    color="#b00020")
    fig.suptitle("Every misclassified test image of the weakest species", fontsize=11)
    fig.tight_layout()
    out = ROOT / "results/figures/weak_classes_errors.png"
    fig.savefig(out, dpi=140)
    print("saved", out)


if __name__ == "__main__":
    main()
