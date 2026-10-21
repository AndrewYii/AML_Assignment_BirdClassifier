"""Grad-CAM figure: rows = models, columns = the same test images (some correct, some wrong).

Usage
    .venv/Scripts/python analysis/gradcam.py e3_convnext_tiny e3_swin_tiny --labels ConvNeXt-T Swin-T --n 6
Images are chosen from the FIRST run's preds.csv: half correct, half misclassified.
"""
import argparse
import textwrap
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.data import build_transforms
from src.gradcam import grad_cam, overlay
from src.models import build_model, norm_stats
from src.utils import ROOT, load_config


def load_run(run, device):
    cfg = load_config(ROOT / "runs" / run / "config.yaml")
    cfg["model"]["pretrained"] = False
    cfg["model"]["grad_ckpt"] = False
    model = build_model(cfg, cfg["model"]["num_classes"]).to(device)
    model.load_state_dict(torch.load(ROOT / "runs" / run / "last.pt", map_location=device)["model"])
    _, tf = build_transforms(cfg["data"]["img_size"], "none", *norm_stats(model), cfg["data"].get("crop_pct", 0.875))
    return model.float(), tf, cfg["data"]["img_size"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--fontsize", type=float, default=7, help="title size; larger titles wrap onto two lines")
    args = ap.parse_args()
    labels = args.labels or args.runs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = json.loads((ROOT / "class_names.json").read_text())
    width = int(22 * 7 / args.fontsize * 1.6)  # characters per title line at this font size
    wrap = lambda t: t if len(t) <= width else "\n".join(textwrap.wrap(t, width))

    preds = pd.read_csv(ROOT / "results" / args.runs[0] / "preds.csv")
    right = preds[preds.target == preds.pred].sample(args.n // 2, random_state=1)
    wrong = preds[preds.target != preds.pred].sample(args.n - args.n // 2, random_state=1)
    picks = pd.concat([right, wrong])

    fig, axes = plt.subplots(len(args.runs) + 1, args.n, figsize=(2.3 * args.n, (2.5 + 0.06 * max(0, args.fontsize - 7)) * (len(args.runs) + 1)))
    for j, row in enumerate(picks.itertuples()):
        img = Image.open(ROOT / "dataset/Test" / row.image).convert("RGB")
        axes[0, j].imshow(img)
        axes[0, j].set_title(wrap(f"true: {names[row.target]}"), fontsize=args.fontsize)
    for i, (run, lab) in enumerate(zip(args.runs, labels), start=1):
        model, tf, size = load_run(run, device)
        for j, row in enumerate(picks.itertuples()):
            img = Image.open(ROOT / "dataset/Test" / row.image).convert("RGB")
            x = tf(img).unsqueeze(0).to(device)
            cam, pred, probs = grad_cam(model, x)
            # same resize+crop as the model input, so the heatmap lines up
            vis = np.array(img.resize((int(size / 0.875 * img.width / min(img.size)),
                                       int(size / 0.875 * img.height / min(img.size)))))
            top = (vis.shape[0] - size) // 2
            left = (vis.shape[1] - size) // 2
            vis = vis[top:top + size, left:left + size]
            axes[i, j].imshow(overlay(vis, cam))
            ok = pred == row.target
            axes[i, j].set_title(wrap(f"{lab}: {names[pred]} ({probs[pred]:.0%})"), fontsize=args.fontsize,
                                 color="green" if ok else "red")
        del model
        torch.cuda.empty_cache()
    for a in axes.ravel():
        a.axis("off")
    fig.tight_layout()
    out = ROOT / "results" / "figures" / f"gradcam_{'_vs_'.join(labels)}.png"
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
