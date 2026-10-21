"""Local Gradio tool: is a run overfitting, underfitting, or a good fit?

Usage
    .venv/Scripts/python analysis/fit_checker_app.py        -> http://127.0.0.1:7861

Tabs
  1. Learning curves  - pick a run: loss / accuracy curves + an automatic diagnosis from its history
  2. Clean check      - load a trained checkpoint and measure accuracy on CLEAN training images
                        (no augmentation, no MixUp) vs the test set. This is the fair train-vs-test gap,
                        because the logged "train accuracy" was measured on augmented / mixed images.
  3. Compare runs     - two runs side by side

How the diagnosis reads a run (rules of thumb, stated in the app):
  overfitting  : big train-test gap (> 10 points) or test loss rising > 15 % above its minimum
  underfitting : the model cannot fit the training data (clean train accuracy < 90 %)
  good fit     : high, stable test accuracy with a small gap
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import random

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data import CUBDataset, build_transforms, read_split
from src.engine import predict
from src.metrics import compute_metrics
from src.models import build_model, norm_stats
from src.utils import ROOT, load_config

TRAIN, TEST = "#1f3cff", "#e8221b"
STYLE = dict(marker="o", markersize=5, linewidth=1.5, alpha=0.85)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def runs_with_history():
    return sorted(p.parent.name for p in (ROOT / "results").glob("*/history.csv"))


def runs_with_checkpoint():
    return sorted(p.parent.name for p in (ROOT / "runs").glob("*/last.pt"))


def describe(run):
    m = ROOT / "results" / run / "metrics.json"
    return json.loads(m.read_text()).get("description", "") if m.exists() else ""


# --------------------------------------------------------------------------- #
# 1. Curves + diagnosis from the logged history
# --------------------------------------------------------------------------- #
def curve_figure(h, title):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    a1.plot(h.epoch, h.train_loss, color=TRAIN, label="Training loss", **STYLE)
    a1.plot(h.epoch, h.test_loss, color=TEST, label="Test loss", **STYLE)
    a1.set(title=f"{title}: loss", xlabel="Epoch", ylabel="Loss")
    a1.set_ylim(bottom=0)
    a2.plot(h.epoch, h.train_acc * 100, color=TRAIN, label="Training accuracy (augmented images)", **STYLE)
    a2.plot(h.epoch, h.test_top1 * 100, color=TEST, label="Test top-1 accuracy", **STYLE)
    a2.set(title=f"{title}: accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    a2.set_ylim(0, 102)
    for a in (a1, a2):
        a.grid(True, color="#d9d9d9")
        a.legend(fontsize=8)
    fig.tight_layout()
    return fig


def diagnose_history(h, run):
    tr, te = h.train_acc.iloc[-1] * 100, h.test_top1.iloc[-1] * 100
    gap = tr - te
    tl = h.test_loss.values
    rise = (tl[-1] - tl.min()) / tl.min() * 100
    best_ep = int(h.test_loss.idxmin()) + 1
    mixed = "mixup" in run or "fullaug" in run or "e6c" in run or "e6f" in run
    lines = [f"**Final epoch:** train {tr:.1f}%, test {te:.1f}%, gap {gap:+.1f} points",
             f"**Test loss:** lowest {tl.min():.3f} at epoch {best_ep}, final {tl[-1]:.3f} ({rise:+.0f}% vs lowest)"]
    if mixed and gap < 0:
        verdict = ("**Train < test because of MixUp/CutMix:** training images are blends of two birds, so they are "
                   "harder than real images. This is not underfitting. Use the *Clean check* tab for a fair gap.")
    elif tr < 90 and not mixed:
        verdict = (f"**Underfitting:** only {tr:.1f}% on its own training images: the model lacks the capacity or "
                   "freedom to fit the data (e.g. a frozen backbone).")
    elif gap > 10:
        verdict = (f"**Overfitting:** a {gap:.0f}-point train-test gap. The model memorises training images much "
                   "better than it generalises (more data, better pretraining or regularisation help).")
    elif rise > 15 and te >= h.test_top1.max() * 100 - 1.0:
        verdict = (f"**Good fit, slightly over-confident:** test accuracy stays at its best level ({te:.1f}%), but test "
                   f"loss rose {rise:.0f}% after epoch {best_ep}. The model grows more confident on the few images it "
                   "gets wrong without making more mistakes: mild overfitting of confidence, not of accuracy.")
    elif rise > 15:
        verdict = (f"**Overfitting:** test loss rose {rise:.0f}% after epoch {best_ep} and test accuracy fell from its "
                   "best level.")
    else:
        verdict = (f"**Good fit:** test accuracy is high ({te:.1f}%) and stable, the gap is small ({gap:.1f} points) "
                   "and the test loss does not rise.")
    lines.append(verdict)
    return "\n\n".join(lines)


def show_run(run):
    if not run:
        return None, "Pick a run."
    h = pd.read_csv(ROOT / "results" / run / "history.csv")
    return curve_figure(h, run), f"_{describe(run)}_\n\n" + diagnose_history(h, run)


# --------------------------------------------------------------------------- #
# 2. Clean check: train vs test accuracy with the SAME (test) transform
# --------------------------------------------------------------------------- #
_CACHE = {}


def load_trained(run):
    if run in _CACHE:
        return _CACHE[run]
    cfg = load_config(ROOT / "runs" / run / "config.yaml")
    cfg["model"]["pretrained"] = False
    cfg["model"]["grad_ckpt"] = False
    model = build_model(cfg, cfg["model"]["num_classes"]).to(DEVICE).eval()
    model.load_state_dict(torch.load(ROOT / "runs" / run / "last.pt", map_location=DEVICE)["model"])
    _, tf = build_transforms(cfg["data"]["img_size"], "none", *norm_stats(model), cfg["data"].get("crop_pct", 0.875))
    _CACHE.clear()  # keep at most one model in memory
    _CACHE[run] = (cfg, model, tf)
    return _CACHE[run]


def accuracy_on(items, img_dir, model, tf, desc):
    ds = CUBDataset(items, img_dir, tf)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    logits, targets = predict(model, loader, DEVICE, desc=desc)
    s, _, _ = compute_metrics(logits, targets)
    return s


def clean_check(run, n_train, progress=gr.Progress()):
    if not run:
        return "Pick a run with a checkpoint.", None
    progress(0.1, desc="loading model")
    cfg, model, tf = load_trained(run)
    train_items = read_split(ROOT / cfg["data"]["train_txt"])
    random.Random(0).shuffle(train_items)
    train_items = train_items[: int(n_train)]
    test_items = read_split(ROOT / cfg["data"]["test_txt"])
    progress(0.3, desc=f"clean training images ({len(train_items)})")
    s_tr = accuracy_on(train_items, ROOT / cfg["data"]["train_dir"], model, tf, "clean train")
    progress(0.65, desc="test images (1,204)")
    s_te = accuracy_on(test_items, ROOT / cfg["data"]["test_dir"], model, tf, "test")
    tr, te = s_tr["top1"] * 100, s_te["top1"] * 100
    gap = tr - te
    if tr < 90:
        verdict = f"**Underfitting:** only {tr:.1f}% on clean training images, so the model cannot fit the training data."
    elif gap > 10:
        verdict = f"**Overfitting:** {gap:.1f}-point gap; the model knows training images much better than new ones."
    elif gap > 5:
        verdict = (f"**Mild overfitting (normal for ~24 images per class):** {gap:.1f}-point gap, with high test "
                   f"accuracy ({te:.1f}%).")
    else:
        verdict = f"**Good fit:** small gap ({gap:.1f} points) and high test accuracy ({te:.1f}%)."
    table = pd.DataFrame({
        "set": [f"clean train ({len(train_items)} images, no augmentation)", "test (1,204 images)"],
        "top-1 (%)": [round(tr, 2), round(te, 2)],
        "mean per-class (%)": [round(s_tr["mean_class_acc"] * 100, 2), round(s_te["mean_class_acc"] * 100, 2)],
        "top-5 (%)": [round(s_tr["top5"] * 100, 2), round(s_te["top5"] * 100, 2)],
    })
    return verdict, table


# --------------------------------------------------------------------------- #
# 3. Compare two runs
# --------------------------------------------------------------------------- #
def compare(a, b):
    if not a or not b:
        return None, "Pick two runs."
    ha = pd.read_csv(ROOT / "results" / a / "history.csv")
    hb = pd.read_csv(ROOT / "results" / b / "history.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey="row")
    for j, (h, name) in enumerate(((ha, a), (hb, b))):
        axes[0, j].plot(h.epoch, h.train_loss, color=TRAIN, label="Training loss", **STYLE)
        axes[0, j].plot(h.epoch, h.test_loss, color=TEST, label="Test loss", **STYLE)
        axes[1, j].plot(h.epoch, h.train_acc * 100, color=TRAIN, label="Training accuracy", **STYLE)
        axes[1, j].plot(h.epoch, h.test_top1 * 100, color=TEST, label="Test top-1", **STYLE)
        axes[0, j].set_title(name, fontsize=10)
        axes[1, j].set_xlabel("Epoch")
        for a_ in axes[:, j]:
            a_.grid(True, color="#d9d9d9")
            a_.legend(fontsize=8)
    axes[0, 0].set_ylabel("Loss")
    axes[1, 0].set_ylabel("Accuracy (%)")
    fig.tight_layout()
    text = f"**{a}**\n\n{diagnose_history(ha, a)}\n\n---\n\n**{b}**\n\n{diagnose_history(hb, b)}"
    return fig, text


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
with gr.Blocks(title="Fit checker") as demo:
    gr.Markdown("# Overfitting / underfitting checker\n"
                "**Overfitting** = training much better than test (large gap, test loss rising). "
                "**Underfitting** = cannot fit even the training data. **Good fit** = high, stable test "
                "accuracy with a small gap.")
    with gr.Tab("1. Learning curves"):
        run_dd = gr.Dropdown(runs_with_history(), label="Run", value="e15b_ema_convnext_base")
        curves = gr.Plot()
        diag = gr.Markdown()
        run_dd.change(show_run, run_dd, [curves, diag])
        demo.load(show_run, run_dd, [curves, diag])
    with gr.Tab("2. Clean train vs test check"):
        gr.Markdown("Measures accuracy on **clean** training images (same pre-processing as the test set: "
                    "no augmentation, no MixUp). This is the fair train-vs-test gap. About 1 minute on the GPU.")
        with gr.Row():
            ck_dd = gr.Dropdown(runs_with_checkpoint(), label="Trained run (checkpoint)",
                                value="e15b_ema_convnext_base")
            n_tr = gr.Slider(200, 4829, value=1204, step=1, label="Number of training images to check")
        go = gr.Button("Run check", variant="primary")
        verdict = gr.Markdown()
        result = gr.Dataframe()
        go.click(clean_check, [ck_dd, n_tr], [verdict, result])
    with gr.Tab("3. Compare two runs"):
        with gr.Row():
            ra = gr.Dropdown(runs_with_history(), label="Run A", value="e15b_ema_convnext_base")
            rb = gr.Dropdown(runs_with_history(), label="Run B", value="e17_convnext_base_ema_fullaug")
        cbtn = gr.Button("Compare", variant="primary")
        cfig = gr.Plot()
        ctext = gr.Markdown()
        cbtn.click(compare, [ra, rb], [cfig, ctext])

if __name__ == "__main__":
    demo.launch(server_port=7861, theme=gr.themes.Soft())
