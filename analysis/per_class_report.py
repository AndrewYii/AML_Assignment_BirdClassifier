"""Per-class accuracy of the final model for all 200 species: a chart + a full table page.

Usage
    .venv/Scripts/python analysis/per_class_report.py
Writes
    results/figures/per_class_accuracy.png   all 200 species sorted by accuracy
    results/final/per_class_table.html       table: worst first, with test images, correct, P/R/F1,
                                             baseline (ResNet-50) accuracy, most-confused species
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd

from src.utils import ROOT

BASELINE_RUN = "e2_baseline_resnet50"


def main():
    names = json.loads((ROOT / "class_names.json").read_text())
    per = pd.read_csv(ROOT / "results/final/per_class.csv")
    preds = pd.read_csv(ROOT / "results/final/preds.csv")
    base = pd.read_csv(ROOT / f"results/{BASELINE_RUN}/per_class.csv").set_index("class_id").accuracy

    per["correct"] = [int(round(a * s)) for a, s in zip(per.accuracy, per.support)]
    per["baseline"] = per.class_id.map(base)
    conf = []
    for c in per.class_id:
        wrong = preds[(preds.target == c) & (preds.pred != c)]
        conf.append(", ".join(f"{names[p]} ×{k}" for p, k in Counter(wrong.pred).most_common(2)))
    per["confused_with"] = conf
    per = per.sort_values(["accuracy", "f1", "class_name"]).reset_index(drop=True)
    per.index += 1

    # ---- chart: all 200 sorted ----
    acc = per.accuracy * 100
    colors = ["#e8221b" if a < 70 else "#eda100" if a < 90 else "#2a78d6" for a in acc]
    fig, ax = plt.subplots(figsize=(14, 4.2))
    ax.bar(range(len(acc)), acc, color=colors, width=0.85)
    ax.axhline(per.accuracy.mean() * 100, color="#111", ls="--", lw=1)
    ax.text(len(acc) - 1, per.accuracy.mean() * 100 - 4, f"average per-class accuracy {per.accuracy.mean()*100:.2f}%",
            fontsize=9, ha="right", bbox=dict(fc="white", ec="none", pad=1))
    n100, n_low = int((acc == 100).sum()), int((acc < 90).sum())
    ax.text(n_low / 2, 30, f"{n_low} species below 90%", ha="center", fontsize=10, color="#111")
    ax.text(n_low + n100 / 2, 50, f"{n100} species at 100%", ha="center", fontsize=11, color="white",
            fontweight="bold")
    ax.set_xlim(-1, len(acc))
    ax.set_ylim(0, 105)
    ax.set_xticks([])
    ax.set_xlabel("200 species, sorted from lowest to highest accuracy")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Per-class accuracy of the final model (red < 70%, orange 70–89%, blue ≥ 90%)", fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "results/figures/per_class_accuracy.png", dpi=180)

    # ---- summary ----
    bands = {"100%": (acc == 100).sum(), "90-99%": ((acc >= 90) & (acc < 100)).sum(),
             "70-89%": ((acc >= 70) & (acc < 90)).sum(), "below 70%": (acc < 70).sum()}
    print("species per accuracy band:", bands)
    print(f"average per-class accuracy {per.accuracy.mean()*100:.2f}%")

    # ---- table page ----
    def pct(x):
        return f"{x*100:.1f}"

    def row_color(a):
        return "#fde2e1" if a < .7 else "#fff3d6" if a < .9 else ""

    rows = "\n".join(
        f"<tr style='background:{row_color(r.accuracy)}'><td>{i}</td><td>{r.class_name}</td><td>{r.support}</td>"
        f"<td>{r.correct}</td><td><b>{pct(r.accuracy)}</b></td><td>{pct(r.precision)}</td><td>{pct(r.recall)}</td>"
        f"<td>{pct(r.f1)}</td><td>{pct(r.baseline)}</td><td>{r.confused_with}</td></tr>"
        for i, r in per.iterrows())
    summary = " · ".join(f"<b>{k}</b>: {v} species" for k, v in bands.items())
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>Per-class accuracy</title><style>
body{{font-family:system-ui,sans-serif;margin:24px auto;max-width:1150px;padding:0 16px;color:#111;background:#fff}}
table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{border:1px solid #ddd;padding:3px 6px;text-align:left}}
th{{background:#f2f2f2;position:sticky;top:0}} img{{max-width:100%}}</style></head><body>
<h1>Per-class accuracy of the final model (200 species)</h1>
<p>Test set: 1,204 images (4–8 per species). Average per-class accuracy <b>{per.accuracy.mean()*100:.2f}%</b>.
{summary}. Rows: red = below 70%, yellow = 70–89%. Sorted worst first.</p>
<img src='../figures/per_class_accuracy.png'>
<table><tr><th>#</th><th>Species</th><th>Test images</th><th>Correct</th><th>Accuracy %</th><th>Precision %</th>
<th>Recall %</th><th>F1 %</th><th>Baseline ResNet-50 acc %</th><th>Mistaken for</th></tr>
{rows}</table></body></html>"""
    (ROOT / "results/final/per_class_table.html").write_text(html, encoding="utf-8")
    per.to_csv(ROOT / "results/final/per_class_sorted.csv")
    print("saved results/final/per_class_table.html, results/figures/per_class_accuracy.png, results/final/per_class_sorted.csv")


if __name__ == "__main__":
    main()
