"""Package a trained run for the Gradio / Hugging Face app.

Usage
    .venv/Scripts/python export_for_app.py --run e5_batch16 --key cnn --label "ConvNeXt-T (CNN)"

What it does
  * saves the weights in half precision (half the file size, same predictions) to app/weights/<key>.pt
  * records label, metrics and config in app/models.json (the app's model dropdown reads this)
  * copies the shared code (src/models.py, src/gradcam.py) and class names into app/
so the app folder is self-contained and can be pushed to a Space as-is.
"""
import argparse
import json
import shutil

import torch

from src.utils import ROOT

APP = ROOT / "app"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--key", required=True, help="short id, e.g. cnn / transformer")
    ap.add_argument("--label", required=True, help="name shown in the app dropdown")
    ap.add_argument("--tta-results", help="optional results dir with TTA metrics to display instead")
    args = ap.parse_args()

    ckpt = torch.load(ROOT / "runs" / args.run / "last.pt", map_location="cpu")
    state = {k: v.half() if v.is_floating_point() else v for k, v in ckpt["model"].items()}
    (APP / "weights").mkdir(parents=True, exist_ok=True)
    torch.save({"model": state, "config": ckpt["config"]}, APP / "weights" / f"{args.key}.pt")

    res = ROOT / "results" / (args.tta_results or args.run)
    metrics = json.loads((res / "metrics.json").read_text())
    registry_path = APP / "models.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    registry[args.key] = {
        "label": args.label,
        "file": f"weights/{args.key}.pt",
        "run": args.run,
        "backbone": ckpt["config"]["model"]["name"],
        "img_size": ckpt["config"]["data"]["img_size"],
        "top1": metrics["top1"],
        "mean_class_acc": metrics["mean_class_acc"],
        "top5": metrics["top5"],
        "macro_f1": metrics["macro_f1"],
        "params_m": metrics.get("params_m"),
    }
    registry_path.write_text(json.dumps(registry, indent=2))

    # shared code + class names -> app/core (the Space has no access to ../src)
    core = APP / "core"
    core.mkdir(exist_ok=True)
    (core / "__init__.py").write_text("")
    for f in ("models.py", "gradcam.py"):
        shutil.copy(ROOT / "src" / f, core / f)
    shutil.copy(ROOT / "class_names.json", APP / "class_names.json")
    print(f"exported {args.run} -> app/weights/{args.key}.pt  ({metrics['top1']:.2%} top-1)")


if __name__ == "__main__":
    main()
