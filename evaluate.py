"""Re-evaluate a trained run (optionally with horizontal-flip test-time augmentation).

Usage
    .venv/Scripts/python evaluate.py --run e2_baseline_resnet50
    .venv/Scripts/python evaluate.py --run e15b_ema_convnext_base --tta   # writes results/e15b_ema_convnext_base_tta/
"""
import argparse

import torch

from src.data import build_loaders, class_names_from_train
from src.engine import predict, save_evaluation
from src.models import build_model, norm_stats
from src.utils import ROOT, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="exp_id under runs/")
    ap.add_argument("--tta", action="store_true", help="average logits of image + flipped image")
    args = ap.parse_args()

    run_dir = ROOT / "runs" / args.run
    cfg = load_config(run_dir / "config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg["model"]["pretrained"] = False  # weights come from the checkpoint
    model = build_model(cfg, cfg["model"]["num_classes"]).to(device)
    model.load_state_dict(torch.load(run_dir / "last.pt", map_location=device)["model"])

    mean, std = norm_stats(model)
    _, test_loader, _ = build_loaders(cfg, mean, std)
    logits, targets = predict(model, test_loader, device, tta=args.tta)
    names = class_names_from_train(ROOT / cfg["data"]["train_txt"])
    out = ROOT / "results" / (args.run + ("_tta" if args.tta else "_eval"))
    summary = save_evaluation(logits, targets, test_loader.dataset.items, names, out)
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in summary.items()})


if __name__ == "__main__":
    main()
