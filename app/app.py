"""Bird species identifier (CUB-200) - Gradio app for Hugging Face Spaces.

Runs the final model of the report: an ensemble of five ConvNeXt-B networks (pretrained on ImageNet-22k,
fine-tuned at 384 x 384 with weight EMA, one per 5-fold cross-validation fold). Their softmax probabilities
are averaged with equal weights - test top-1 94.93 %, average per-class accuracy 94.88 %.

One page: upload / webcam / paste a photo -> top-5 species with confidence, verdict, Grad-CAM heatmap
(the average of the five members' heatmaps).

The weights (fp16 .pt) are listed in models.json (written by export_for_app.py).
Runs on a Hugging Face ZeroGPU Space (GPU attached only while `identify` runs, via @spaces.GPU)
and on a normal CPU machine (locally, where the `spaces` package is absent).
"""
import json
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from core.gradcam import grad_cam, overlay
from core.models import build_model, norm_stats

# ZeroGPU: the Space gets a GPU only inside functions decorated with @spaces.GPU.
# Locally the `spaces` package does not exist, so the decorator becomes a no-op and we run on CPU.
try:
    import spaces

    gpu = spaces.GPU(duration=60)
    DEVICE = torch.device("cuda")
except ImportError:
    def gpu(fn):
        return fn

    DEVICE = torch.device("cpu")

HERE = Path(__file__).resolve().parent
CLASS_NAMES = json.loads((HERE / "class_names.json").read_text())
REGISTRY = json.loads((HERE / "models.json").read_text())
torch.set_num_threads(max(1, torch.get_num_threads()))


# --------------------------------------------------------------------------- #
# Load the final model (once, at start-up)
# --------------------------------------------------------------------------- #
def load(entry):
    ckpt = torch.load(HERE / entry["file"], map_location="cpu")
    cfg = ckpt["config"]
    cfg["model"]["pretrained"] = False  # weights come from our checkpoint
    cfg["model"]["grad_ckpt"] = False
    model = build_model(cfg, len(CLASS_NAMES))
    model.load_state_dict({k: v.float() for k, v in ckpt["model"].items()})
    model.eval().to(DEVICE)  # on ZeroGPU this is allowed at start-up; the GPU is attached during @gpu calls
    size = cfg["data"]["img_size"]
    crop_pct = cfg["data"].get("crop_pct", 0.875)
    mean, std = norm_stats(model)
    # exactly the test-time pre-processing used for the reported results
    geometry = T.Compose([T.Resize(int(size / crop_pct), interpolation=T.InterpolationMode.BICUBIC),
                          T.CenterCrop(size)])
    to_tensor = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    return {"model": model, "geometry": geometry, "to_tensor": to_tensor, **entry}


MEMBERS = [load(REGISTRY[k]) for k in sorted(REGISTRY)]  # fold1 ... fold5, identical pre-processing


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def final_model(img, with_cam):
    """Top-5 {species: probability} and an optional Grad-CAM overlay for one photo.
    The ensemble prediction is the plain average of the five members' softmax probabilities."""
    crop = MEMBERS[0]["geometry"](img.convert("RGB"))
    x = MEMBERS[0]["to_tensor"](crop).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.stack([m["model"](x).softmax(1)[0].float().cpu() for m in MEMBERS]).mean(0)
    heat = None
    if with_cam:
        # every member explains the ensemble's answer; the five heatmaps are averaged
        top1 = int(probs.argmax())
        cams = [grad_cam(m["model"], x, class_idx=top1)[0] for m in MEMBERS]
        heat = Image.fromarray(overlay(np.array(crop), np.mean(cams, axis=0)))
    top = torch.topk(probs.float().cpu(), 5)
    return {CLASS_NAMES[i]: float(p) for p, i in zip(top.values.tolist(), top.indices.tolist())}, heat


def verdict_html(scores):
    """Short, honest reading of the prediction (confident / close call / unsure)."""
    (n1, p1), (n2, p2) = list(scores.items())[:2]
    wiki = f"https://en.wikipedia.org/w/index.php?search={n1.replace(' ', '+')}"
    if p1 >= 0.6 and p1 - p2 >= 0.3:
        tone, msg = "ok", "Confident prediction."
    elif p1 - p2 < 0.15:
        tone, msg = "warn", (f"Close call with <b>{n2}</b> ({p2:.0%}). These species look alike, so check the "
                             "bill, wing bars and eye markings.")
    else:
        tone, msg = "warn", "Moderate confidence. A clearer, closer photo of the bird usually helps."
    return (f"<div class='verdict {tone}'><div class='species'>{n1}</div>"
            f"<div class='conf'>{p1:.1%} confidence</div><div class='msg'>{msg}</div>"
            f"<a href='{wiki}' target='_blank'>Read about this species ↗</a></div>")


@gpu
def identify(img, with_cam):
    if img is None:
        return None, None, "<div class='verdict'>Upload a bird photo to start.</div>"
    scores, heat = final_model(img, with_cam)
    return scores, heat, verdict_html(scores)


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
EXAMPLES = sorted(str(p) for p in (HERE / "examples").glob("*.jpg"))

CSS = """
.hero h1 {margin-bottom: 0.2rem}
.hero p {margin-top: 0; opacity: 0.8}
.verdict {padding: 14px 16px; border-radius: 12px; border: 1px solid var(--border-color-primary);}
.verdict .species {font-size: 1.4rem; font-weight: 700}
.verdict .conf {font-size: 1rem; opacity: 0.8; margin-bottom: 6px}
.verdict.ok {border-left: 6px solid #1baf7a}
.verdict.warn {border-left: 6px solid #eda100}
.verdict a {font-size: 0.9rem}
"""

with gr.Blocks(title="Bird Species Identifier - CUB-200") as demo:
    gr.Markdown(
        "# 🐦 Bird Species Identifier\n"
        "Recognises **200 bird species** (Caltech-UCSD Birds-200). Upload a photo: the model returns its five "
        "most likely species and a heatmap of where it looked.",
        elem_classes="hero",
    )
    with gr.Row():
        with gr.Column(scale=5):
            image_in = gr.Image(type="pil", label="Bird photo", sources=["upload", "webcam", "clipboard"],
                                height=360)
            cam_cb = gr.Checkbox(value=True, label="Show Grad-CAM heatmap (where the model looked)")
            go = gr.Button("Identify species", variant="primary")
            if EXAMPLES:
                gr.Examples(EXAMPLES, inputs=image_in, label="Try a test-set photo")
            gr.Markdown("**Tips:** a clear, close photo with the whole bird visible works best. "
                        "The model knows the 200 CUB-200 species (mostly North American birds).")
        with gr.Column(scale=5):
            verdict = gr.HTML("<div class='verdict'>Upload a bird photo to start.</div>")
            label_out = gr.Label(num_top_classes=5, label="Top-5 species")
            cam_out = gr.Image(label="Grad-CAM: red = regions that drove the prediction", height=300)
    for trigger in (go.click, image_in.upload, cam_cb.change):
        trigger(identify, [image_in, cam_cb], [label_out, cam_out, verdict])


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="blue"), css=CSS)
