"""Stratified k-fold split of the TRAINING set only (the test set is never touched), plus one config per fold.

Why: there is no validation split in the main protocol, and choosing between experiments by test accuracy
can make the reported test number slightly optimistic. Cross-validation on the training images gives an
estimate that never looks at the test set.

Usage
    .venv/Scripts/python analysis/make_cv_folds.py --k 5
Writes
    dataset/cv5/fold{i}_train.txt, dataset/cv5/fold{i}_val.txt          (same "name label" format)
    configs/cv/cv_final_f{i}.yaml     final recipe (= e15b_ema_convnext_base), validated on fold i
    configs/cv/cv_baseline_f{i}.yaml  baseline recipe (= e2_baseline_resnet50), validated on fold i
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.model_selection import StratifiedKFold

from src.data import read_split
from src.utils import ROOT

RECIPES = {"final": "e15b_ema_convnext_base.yaml", "baseline": "e2_baseline_resnet50.yaml"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    items = read_split(ROOT / "dataset/train.txt")
    names, labels = [f for f, _ in items], [y for _, y in items]
    out = ROOT / "dataset" / f"cv{args.k}"
    out.mkdir(parents=True, exist_ok=True)
    cfg_dir = ROOT / "configs" / "cv"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)
    for i, (tr, va) in enumerate(skf.split(names, labels), start=1):
        for part, idx in (("train", tr), ("val", va)):
            (out / f"fold{i}_{part}.txt").write_text(
                "".join(f"{names[j]} {labels[j]}\n" for j in sorted(idx)), encoding="utf-8")
        for key, base in RECIPES.items():
            (cfg_dir / f"cv_{key}_f{i}.yaml").write_text(
                f"base: ../{base}\n"
                f"exp_id: cv{args.k}_{key}_f{i}\n"
                f'description: "{args.k}-fold CV, fold {i}: the {key} recipe trained on the other {args.k - 1} folds of the '
                f'training set and validated on fold {i} (test set untouched)"\n'
                + ("save_ckpt: true   # kept: the 5 fold models are also evaluated as a k-fold ensemble on the test set\n"
                   if key == "final" else "save_ckpt: false\n")
                + "data:\n"
                f"  train_txt: dataset/cv{args.k}/fold{i}_train.txt\n"
                f"  test_txt: dataset/cv{args.k}/fold{i}_val.txt\n"
                "  test_dir: dataset/Train\n", encoding="utf-8")
        per_class = min(sum(1 for j in va if labels[j] == c) for c in set(labels))
        print(f"fold {i}: {len(tr)} train / {len(va)} val images (min {per_class} val images per species)")
    print(f"wrote {out} and {cfg_dir}")


if __name__ == "__main__":
    main()
