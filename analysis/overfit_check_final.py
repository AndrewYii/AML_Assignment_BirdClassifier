"""Is the FINAL model (single ConvNeXt-B, EMA weights, no TTA) overfitting?

Runs the exact final prediction pipeline on ALL 4,829 clean training images (test transform: no augmentation)
and on the 1,204 test images, then compares accuracy and loss.

Usage:  .venv/Scripts/python analysis/overfit_check_final.py   -> results/final/overfit_check.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import CUBDataset, build_transforms, read_split
from src.engine import predict
from src.models import build_model, norm_stats
from src.utils import ROOT, load_config, save_json

FINAL_RUN = "e15b_ema_convnext_base"


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def probs_on(model, tf, items, img_dir, device):
    loader = DataLoader(CUBDataset(items, img_dir, tf), batch_size=16, num_workers=2)
    logits, targets = predict(model, loader, device, desc=img_dir.name)
    return softmax(logits), targets


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(ROOT / "runs" / FINAL_RUN / "config.yaml")
    cfg["model"]["pretrained"] = False
    cfg["model"]["grad_ckpt"] = False
    model = build_model(cfg, 200).to(device).eval()
    model.load_state_dict(torch.load(ROOT / "runs" / FINAL_RUN / "last.pt", map_location=device)["model"])
    _, tf = build_transforms(cfg["data"]["img_size"], "none", *norm_stats(model), cfg["data"].get("crop_pct", 0.875))
    out = {"run": FINAL_RUN}
    for split, txt, d in (("train", "train.txt", "Train"), ("test", "test.txt", "Test")):
        p, y = probs_on(model, tf, read_split(ROOT / "dataset" / txt), ROOT / "dataset" / d, device)
        pc = np.array([(p.argmax(1)[y == c] == c).mean() for c in range(200)])
        out[f"{split}_acc"] = float((p.argmax(1) == y).mean())
        out[f"{split}_loss"] = float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())
        out[f"{split}_classes_at_100"] = int((pc == 1).sum())
        out[f"{split}_worst_class_acc"] = float(pc.min())
    out["gap_points"] = (out["train_acc"] - out["test_acc"]) * 100
    save_json(out, ROOT / "results/final/overfit_check.json")
    for k, v in out.items():
        print(f"{k:30s} {v:.4f}" if isinstance(v, float) else f"{k:30s} {v}")


if __name__ == "__main__":
    main()
