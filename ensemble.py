"""Test-time augmentation and ensembles of trained runs, always evaluated on the official test set.

Usage
    # k-fold ensemble: the 5 cross-validation models of the final recipe (each trained on 4/5 of the training set)
    .venv/Scripts/python ensemble.py --runs cv5_final_f1 cv5_final_f2 cv5_final_f3 cv5_final_f4 cv5_final_f5 \
        --label "5-fold ensemble of ConvNeXt-B" --out kfold_ensemble

Rules fixed BEFORE looking at test results (so nothing is tuned on the test set):
  * each model is the LAST-epoch checkpoint of its run
  * TTA = average of the logits of the image and its horizontal mirror (birds face either way)
  * ensemble = plain average of the members' softmax probabilities, equal weights
Writes
  results/<run>/logits[_tta].npy             cached test logits per member
  results/<run>_tta/metrics.json ...         each member with TTA alone (for the ablation table)
  results/final/metrics.json, preds.csv ...  the ensemble
"""
import argparse

import numpy as np
import torch

from src.data import build_loaders, class_names_from_train
from src.engine import predict, save_evaluation
from src.models import build_model, norm_stats
from src.utils import ROOT, load_config, save_json


def member_logits(run, tta, device):
    """Test logits of one trained run (cached to disk)."""
    cache = ROOT / "results" / run / f"logits{'_tta' if tta else ''}.npy"
    cfg = load_config(ROOT / "runs" / run / "config.yaml")
    cfg["model"]["pretrained"] = False
    cfg["model"]["grad_ckpt"] = False  # inference only
    cfg["data"]["num_workers"] = 2
    # always evaluate on the official test set (cross-validation runs were validated on a training fold)
    cfg["data"]["test_txt"], cfg["data"]["test_dir"] = "dataset/test.txt", "dataset/Test"
    if cache.exists():
        logits = np.load(cache)
        _, loader, _ = build_loaders(cfg, (0, 0, 0), (1, 1, 1))  # only for targets/items
        targets = np.array(loader.dataset.targets)
        return logits, targets, loader.dataset.items
    model = build_model(cfg, cfg["model"]["num_classes"]).to(device)
    model.load_state_dict(torch.load(ROOT / "runs" / run / "last.pt", map_location=device)["model"])
    _, loader, _ = build_loaders(cfg, *norm_stats(model))
    logits, targets = predict(model, loader, device, tta=tta, desc=f"{run}{' +TTA' if tta else ''}")
    np.save(cache, logits)
    del model
    torch.cuda.empty_cache()
    return logits, targets, loader.dataset.items


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--label", default="ensemble")
    ap.add_argument("--out", default="final")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = class_names_from_train(ROOT / "dataset/train.txt")

    probs, targets, items = [], None, None
    for run in args.runs:
        logits, t, it = member_logits(run, args.tta, device)
        assert targets is None or (t == targets).all(), "members must be evaluated on the same test order"
        targets, items = t, it
        if args.tta:  # each member with TTA alone, for the ablation table
            s = save_evaluation(logits, targets, items, names, ROOT / "results" / f"{run}_tta")
            print(f"{run} +TTA: top1 {s['top1']:.4f}  mean-class {s['mean_class_acc']:.4f}")
        probs.append(softmax(logits))

    avg = np.mean(probs, axis=0)
    # save_evaluation expects logits; log-probabilities give the same argmax / ranking
    summary = save_evaluation(np.log(avg + 1e-12), targets, items, names, ROOT / "results" / args.out)
    summary.update({"label": args.label, "members": args.runs, "tta": args.tta})
    save_json(summary, ROOT / "results" / args.out / "metrics.json")
    print(f"ENSEMBLE {args.label}: top1 {summary['top1']:.4f}  mean-class {summary['mean_class_acc']:.4f}  "
          f"top5 {summary['top5']:.4f}  macroF1 {summary['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
