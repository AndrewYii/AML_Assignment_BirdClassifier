"""t-SNE of penultimate-layer features on the test set (qualitative result).

Usage
    .venv/Scripts/python analysis/tsne.py e2_baseline_resnet50 e15b_ema_convnext_base --labels "Baseline ResNet-50" "Single ConvNeXt-B"

For readability only 20 randomly chosen classes (fixed seed) are plotted;
the same classes are used for every model so the panels are comparable.
Also reports the silhouette score on ALL test features (higher = tighter, better-separated clusters).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from src.data import build_loaders, class_names_from_train
from src.models import build_model, extract_features, norm_stats
from src.utils import ROOT, load_config, save_json


@torch.no_grad()
def features_for(run, device):
    run_dir = ROOT / "runs" / run
    cfg = load_config(run_dir / "config.yaml")
    cfg["model"]["pretrained"] = False
    cfg["model"]["grad_ckpt"] = False
    cfg["data"]["num_workers"] = 2
    model = build_model(cfg, cfg["model"]["num_classes"]).to(device).eval()
    model.load_state_dict(torch.load(run_dir / "last.pt", map_location=device)["model"])
    _, loader, _ = build_loaders(cfg, *norm_stats(model))
    feats, ys = [], []
    for x, y in loader:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            f = extract_features(model, x.to(device)).float()
        feats.append(f.cpu())
        ys.append(y)
    return torch.cat(feats).numpy(), torch.cat(ys).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--n-classes", type=int, default=20)
    args = ap.parse_args()
    labels = args.labels or args.runs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = class_names_from_train(ROOT / "dataset/train.txt")

    rng = np.random.default_rng(0)
    chosen = rng.choice(len(names), args.n_classes, replace=False)

    fig, axes = plt.subplots(1, len(args.runs), figsize=(5.2 * len(args.runs), 5))
    axes = np.atleast_1d(axes)
    scores = {}
    for ax, run, lab in zip(axes, args.runs, labels):
        X, y = features_for(run, device)
        scores[run] = float(silhouette_score(X, y, metric="cosine"))
        mask = np.isin(y, chosen)
        Z = TSNE(n_components=2, perplexity=15, init="pca", random_state=0).fit_transform(X[mask])
        cmap = plt.get_cmap("tab20")
        for i, c in enumerate(chosen):
            sel = y[mask] == c
            ax.scatter(Z[sel, 0], Z[sel, 1], s=14, color=cmap(i % 20), label=names[c])
        ax.set_title(f"{lab}\nsilhouette (all 200 classes) = {scores[run]:.3f}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[-1].legend(fontsize=6, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    slug = "_vs_".join(l.lower().replace(" ", "-") for l in labels)  # no spaces in file names
    out = ROOT / "results" / "figures" / f"tsne_{slug}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    save_json(scores, out.with_suffix(".json"))
    print("saved", out, scores)


if __name__ == "__main__":
    main()
