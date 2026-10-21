"""Regenerate every figure of the report in one go (consistent style, one dot per epoch).

Usage
    .venv/Scripts/python analysis/make_figures.py
Figures whose runs are missing are skipped with a note.
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

from src.utils import ROOT

PY = sys.executable
FIG = ROOT / "results" / "figures"

# (name, [(run, label), ...]) -> curves_<name>.png and compare_<name>.png
CURVE_GROUPS = [
    ("e2_baseline", [("e2_baseline_resnet50", "ResNet-50 baseline")]),
    ("e3_backbones", [("e2_baseline_resnet50", "ResNet-50"), ("e3_convnext_tiny", "ConvNeXt-T"),
                      ("e3_swin_tiny", "Swin-T"), ("e3_deit_small", "DeiT-S")]),
    ("e4_optimizers", [("e4_sgd", "SGD+momentum"), ("e4_adam", "Adam"), ("e3_convnext_tiny", "AdamW")]),
    ("e5_batch", [("e5_batch16", "Batch 16"), ("e3_convnext_tiny", "Batch 32"), ("e5_batch64", "Batch 64")]),
    ("e6_regularisers", [("e6a_strong_aug", "+Strong aug"), ("e6b_label_smoothing", "+Label smooth"),
                         ("e6c_mixup_cutmix", "+MixUp/CutMix"), ("e6d_drop_path", "+Drop-path 0.2"),
                         ("e6e_weight_decay", "+WD 0.2")]),
    ("e6_long", [("e6g_no_reg_long", "60 ep, no extra reg."), ("e6f_strong_aug_mix_long", "60 ep, aug + MixUp")]),
    ("e7_pooling", [("e5_batch16", "Average (ref.)"), ("e7_pool_max", "Max"), ("e7_pool_min", "Min"),
                    ("e7_pool_gem", "GeM")]),
    ("e8_finetune", [("e8_linear_probe", "Linear probe"), ("e8_uniform_low", "One LR (low)"),
                     ("e8_uniform_high", "One LR (high)"), ("e8_layer_decay", "Layer-wise decay"),
                     ("e5_batch16", "Differential (ref.)")]),
    ("e10_resolution", [("e5_batch16", "ConvNeXt-T 224px"), ("e10_res384", "ConvNeXt-T 384px")]),
    ("e11_scaling", [("e10_res384", "ConvNeXt-T IN-1k"), ("e11_convnext_base_22k_384", "ConvNeXt-B IN-22k")]),
    ("final_cnn_vs_vit", [("e15b_ema_convnext_base", "ConvNeXt-B + EMA (CNN)"),
                          ("e18_swin_base_22k_384_ema", "Swin-B + EMA (Transformer)")]),
    ("baseline_vs_best", [("e2_baseline_resnet50", "Baseline ResNet-50"), ("e15b_ema_convnext_base", "Final CNN ConvNeXt-B + EMA")]),
    ("e15a_ema", [("e5_batch16", "Without EMA"), ("e15a_ema_tiny", "With EMA")]),
    ("e15b_ema", [("e11_convnext_base_22k_384", "ConvNeXt-B, no EMA"), ("e15b_ema_convnext_base", "ConvNeXt-B, EMA")]),
    ("e14_two_phase", [("e5_batch16", "One phase (ref.)"), ("e14_two_phase", "Two phase (8 frozen + 22)")]),
]


def done(run):
    return (ROOT / "results" / run / "history.csv").exists()


def top1(run):
    return json.loads((ROOT / "results" / run / "metrics.json").read_text())["top1"] * 100


def bar_style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x" if ax.get_xlabel() else "y", color="#e4e3df")
    ax.set_axisbelow(True)


def regulariser_deltas():
    # reference = 3-seed mean of the same recipe without extra regularisation (see the noise check)
    seeds = [top1(r) for r in ("e5_batch16", "e5_batch16_seed1", "e5_batch16_seed2")]
    ref = sum(seeds) / len(seeds)
    runs = [("Strong aug", "e6a_strong_aug"), ("Label smoothing", "e6b_label_smoothing"),
            ("MixUp/CutMix", "e6c_mixup_cutmix"), ("Drop-path 0.2", "e6d_drop_path"), ("Weight decay 0.2", "e6e_weight_decay")]
    vals = [(n, top1(r) - ref) for n, r in runs if done(r)]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    names, ds = [n for n, _ in vals], [d for _, d in vals]
    bars = ax.barh(names, ds, color=["#eb6834" if d < -1.3 else "#9a9893" for d in ds], height=0.6)
    ax.axvspan(-0.66, 0.66, color="#2a78d6", alpha=0.08, label="±1 std seed noise")
    ax.axvline(0, color="#0b0b0b", lw=1)
    for b, d in zip(bars, ds):
        ax.text(d - 0.05 if d < 0 else d + 0.05, b.get_y() + b.get_height() / 2, f"{d:+.2f}", va="center",
                ha="right" if d < 0 else "left", fontsize=9)
    ax.set_xlim(min(ds) - 0.6, 1.0)
    ax.set_xlabel(f"Δ test top-1 (points) vs no extra regularisation ({ref:.2f}%, 3-seed mean)")
    ax.set_title("Each regulariser added alone (30 epochs, ConvNeXt-T)", fontsize=10)
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    bar_style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "e6_regulariser_deltas.png", dpi=200)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    for name, items in CURVE_GROUPS:
        items = [(r, l) for r, l in items if done(r)]
        if not items:
            print(f"skip {name} (no runs yet)")
            continue
        subprocess.run([PY, str(ROOT / "analysis/plot_curves.py"), *[r for r, _ in items], "--name", name,
                        "--labels", *[l for _, l in items]], check=True, stdout=subprocess.DEVNULL)
        print("curves", name)
    regulariser_deltas()
    for script in ("story_chart.py", "data_size.py", "arch_diagram.py"):
        subprocess.run([PY, str(ROOT / "analysis" / script)], check=True, stdout=subprocess.DEVNULL)
    subprocess.run([PY, str(ROOT / "analysis/plot_report_curves.py"), "e2_baseline_resnet50", "e15b_ema_convnext_base",
                    "--labels", "Baseline ResNet-50", "Single ConvNeXt-B", "--name", "baseline_vs_final",
                    "--metric", "both"], check=True, stdout=subprocess.DEVNULL)
    subprocess.run([PY, str(ROOT / "analysis/plot_report_curves.py"), "e2_baseline_resnet50", "e15b_ema_convnext_base",
                    "--labels", "Baseline ResNet-50", "Single ConvNeXt-B", "--name", "baseline_vs_final",
                    "--metric", "row"], check=True, stdout=subprocess.DEVNULL)
    report_pairs()
    print("bar charts, progression, cnn_vs_vit, data size, architecture, report figures: done")


def report_pairs():
    """Two-panel report figures as single images (a PDF printer never splits them from their caption)."""
    from PIL import Image

    def pair(a, b, out, h=900, gap=40):
        ims = [Image.open(FIG / x).convert("RGB") for x in (a, b)]
        ims = [im.resize((int(im.width * h / im.height), h), Image.LANCZOS) for im in ims]
        canvas = Image.new("RGB", (ims[0].width + ims[1].width + gap, h), "white")
        canvas.paste(ims[0], (0, 0))
        canvas.paste(ims[1], (ims[0].width + gap, 0))
        canvas.save(FIG / out, optimize=True)

    pair("e6_regulariser_deltas.png", "e9_data_size.png", "report_fig_reg_data.png")


if __name__ == "__main__":
    main()
