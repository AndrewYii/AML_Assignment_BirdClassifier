"""Train one experiment end to end.

Usage
    .venv/Scripts/python train.py --config configs/e2_baseline_resnet50.yaml
    .venv/Scripts/python train.py --config configs/e2_baseline_resnet50.yaml --set train.epochs=1

Outputs
    runs/<exp_id>/      config.yaml, history.csv, last.pt   (git-ignored, large)
    results/<exp_id>/   metrics.json, per_class.csv, preds.csv, confusion.npy, history.csv, config.yaml
"""
import argparse
import shutil

import torch

from src.data import build_loaders, class_names_from_train
from src.engine import predict, run_training, save_evaluation
from src.models import build_model, norm_stats
from src.utils import (ROOT, Timer, count_params, git_hash, load_config,
                       measure_flops_and_latency, save_json, save_yaml, set_seed)


def plot_history(history, title, out_path):
    """Save train-vs-test loss/accuracy curves for this run (overfitting shows up as a widening gap)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    train_c, test_c = "#2a78d6", "#eb6834"  # same bright palette as analysis/plot_curves.py
    a1.plot(ep, [h["train_loss"] for h in history], color=train_c, lw=2, marker="o", ms=3, label="train")
    a1.plot(ep, [h.get("test_loss") for h in history], color=test_c, lw=2, marker="o", ms=3, label="test")
    a2.plot(ep, [h["train_acc"] * 100 for h in history], color=train_c, lw=2, marker="o", ms=3, label="train")
    a2.plot(ep, [(h.get("test_top1") or 0) * 100 for h in history], color=test_c, lw=2, marker="o", ms=3, label="test top-1")
    a1.set(xlabel="epoch", ylabel="loss", title=f"{title} - loss")
    a2.set(xlabel="epoch", ylabel="accuracy (%)", title=f"{title} - accuracy")
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="override keys, e.g. train.epochs=1")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    cfg["git"] = git_hash()
    set_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = ROOT / "runs" / cfg["exp_id"]
    res_dir = ROOT / "results" / cfg["exp_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, run_dir / "config.yaml")

    # ---- model first: its pretraining normalisation (mean/std) feeds the transforms ----
    num_classes = cfg["data"]["subset"]["n_classes"] if (cfg["data"].get("subset") or {}).get("n_classes") \
        else cfg["model"]["num_classes"]
    model = build_model(cfg, num_classes).to(device)
    mean, std = norm_stats(model)
    train_loader, test_loader, _ = build_loaders(cfg, mean, std)
    print(f"[{cfg['exp_id']}] {cfg['model']['name']}  params {count_params(model):.1f}M  "
          f"train {len(train_loader.dataset)}  test {len(test_loader.dataset)}  device {device}", flush=True)

    # ---- train for the fixed epoch budget ----
    with Timer() as t:
        # with EMA enabled, the returned model is the weight-averaged copy: that is what we save and report
        history, model = run_training(model, train_loader, test_loader, cfg, num_classes, device,
                                      log_path=run_dir / "history.csv")

    if cfg.get("save_ckpt", True):
        torch.save({"model": model.state_dict(), "config": cfg}, run_dir / "last.pt")

    # ---- final evaluation of the LAST epoch (no checkpoint selection on test) ----
    logits, targets = predict(model, test_loader, device)
    names = class_names_from_train(ROOT / cfg["data"]["train_txt"])
    if num_classes != len(names):  # tiny-subset sanity run
        names = None
    summary = save_evaluation(logits, targets, test_loader.dataset.items, names, res_dir)
    gflops, latency = measure_flops_and_latency(model, cfg["data"]["img_size"], device)
    summary.update({
        "exp_id": cfg["exp_id"], "model": cfg["model"]["name"], "img_size": cfg["data"]["img_size"],
        "params_m": count_params(model), "gflops": gflops, "latency_ms": latency,
        "train_time_min": t.elapsed / 60, "epochs": cfg["train"]["epochs"],
        "final_train_acc": history[-1]["train_acc"], "final_train_loss": history[-1]["train_loss"],
        "final_test_loss": history[-1].get("test_loss"),
        # train-test gap: the overfitting indicator used throughout the report
        "train_test_gap": history[-1]["train_acc"] - summary["top1"],
        "description": cfg.get("description", ""),
    })
    save_json(summary, res_dir / "metrics.json")
    save_yaml(cfg, res_dir / "config.yaml")
    shutil.copy(run_dir / "history.csv", res_dir / "history.csv")
    plot_history(history, cfg["exp_id"], res_dir / "curves.png")
    print(f"[{cfg['exp_id']}] FINAL top1 {summary['top1']:.4f}  mean-class {summary['mean_class_acc']:.4f}  "
          f"top5 {summary['top5']:.4f}  macroF1 {summary['macro_f1']:.4f}  ({t.elapsed/60:.1f} min)", flush=True)


if __name__ == "__main__":  # required on Windows for DataLoader worker processes
    main()
