"""Summarise k-fold cross-validation runs and compare them with the test-set result.

Usage
    .venv/Scripts/python analysis/cv_summary.py --k 5
Writes
    results/cv5_summary.json   per-recipe mean / std of validation top-1 and per-class accuracy, the last-epoch
                               vs best-epoch gap, and the test result of the same recipe
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.utils import ROOT, save_json

TEST_RUN = {"final": "e15b_ema_convnext_base", "baseline": "e2_baseline_resnet50"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    out = {}
    for key, test_run in TEST_RUN.items():
        rows = []
        for i in range(1, args.k + 1):
            d = ROOT / "results" / f"cv{args.k}_{key}_f{i}"
            if not (d / "metrics.json").exists():
                continue
            m = json.loads((d / "metrics.json").read_text())
            h = pd.read_csv(d / "history.csv")
            rows.append({"fold": i, "top1": m["top1"] * 100, "mca": m["mean_class_acc"] * 100,
                         "best_epoch_top1": h.test_top1.max() * 100, "best_epoch": int(h.test_top1.idxmax()) + 1,
                         "train_acc": h.train_acc.iloc[-1] * 100})
        if not rows:
            continue
        df = pd.DataFrame(rows)
        test = json.loads((ROOT / "results" / test_run / "metrics.json").read_text())
        out[key] = {
            "folds_done": len(df), "per_fold": df.round(2).to_dict("records"),
            "val_top1_mean": df.top1.mean(), "val_top1_std": df.top1.std(ddof=1) if len(df) > 1 else None,
            "val_mca_mean": df.mca.mean(),
            # how much a "pick the best epoch" rule would have gained over our fixed last-epoch rule
            "best_minus_last_mean": (df.best_epoch_top1 - df.top1).mean(),
            "test_top1": test["top1"] * 100, "test_run": test_run,
        }
        print(f"{key:8s}: {len(df)}/{args.k} folds  val top-1 {df.top1.mean():.2f} ± "
              f"{(df.top1.std(ddof=1) if len(df) > 1 else float('nan')):.2f}  (test {test['top1'] * 100:.2f}); "
              f"best-epoch minus last-epoch {np.mean(df.best_epoch_top1 - df.top1):.2f}")
    save_json(out, ROOT / "results" / f"cv{args.k}_summary.json")


if __name__ == "__main__":
    main()
