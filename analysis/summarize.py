"""Collect every results/<exp_id>/metrics.json into one table for the report.

Outputs results/summary.csv and results/summary.md (markdown table, sorted by experiment id).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.utils import ROOT

COLS = ["exp_id", "model", "img_size", "top1", "mean_class_acc", "top5", "macro_f1",
        "final_train_acc", "train_test_gap", "params_m", "gflops", "train_time_min", "description"]


def main():
    rows = []
    for f in sorted((ROOT / "results").glob("*/metrics.json")):
        m = json.loads(f.read_text())
        m["exp_id"] = f.parent.name  # the results folder (copies such as results/final_single_* keep their own name)
        rows.append(m)
    df = pd.DataFrame(rows)
    df = df[[c for c in COLS if c in df.columns]].sort_values("exp_id")
    df.to_csv(ROOT / "results" / "summary.csv", index=False)

    show = df.copy()
    for c in ["top1", "mean_class_acc", "top5", "macro_f1", "final_train_acc", "train_test_gap"]:
        if c in show:
            show[c] = (show[c] * 100).round(2)
    for c in ["params_m", "gflops", "train_time_min"]:
        if c in show:
            show[c] = show[c].astype(float).round(1)
    md = show.drop(columns=["description"], errors="ignore").to_markdown(index=False)
    (ROOT / "results" / "summary.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
