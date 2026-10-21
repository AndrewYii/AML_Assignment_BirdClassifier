"""Grad-CAM for CNNs and transformers (shared by analysis/gradcam.py and the Gradio app).

Grad-CAM: take the feature map A of the last stage, the gradient G of the class
score w.r.t. A, weight each channel by its mean gradient and keep the positive part:
    cam = ReLU( sum_c mean_hw(G_c) * A_c )
It shows which image regions pushed the model towards its prediction.
"""
import numpy as np
import torch
import torch.nn.functional as F


def target_layer(model):
    """Last feature stage for each architecture family used in this project."""
    for attr in ("stages", "layers"):  # ConvNeXt (stages), Swin (layers)
        if hasattr(model, attr):
            return getattr(model, attr)[-1]
    if hasattr(model, "layer4"):  # ResNet
        return model.layer4
    if hasattr(model, "blocks"):  # ViT / DeiT
        return model.blocks[-1].norm1
    raise ValueError("unknown architecture for Grad-CAM")


def _to_nchw(t, model):
    """Bring activations/gradients to (B, C, H, W) whatever the architecture layout."""
    if t.ndim == 3:  # ViT tokens (B, N, C): drop class/register tokens, reshape the patch grid
        t = t[:, getattr(model, "num_prefix_tokens", 1):]
        side = int(round(t.shape[1] ** 0.5))
        return t.transpose(1, 2).reshape(t.shape[0], -1, side, side)
    if t.ndim == 4 and "swin" in model.__class__.__name__.lower():  # Swin is channels-last
        return t.permute(0, 3, 1, 2)
    return t


def grad_cam(model, x, class_idx=None):
    """Return (cam [H, W] in 0..1 at input resolution, predicted class, probabilities)."""
    store = {}
    layer = target_layer(model)
    # hooks catch the feature map on the way forward and its gradient on the way back
    h1 = layer.register_forward_hook(lambda m, i, o: store.__setitem__("act", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("grad", go[0]))
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            x = x.requires_grad_(True)
            logits = model(x)
            probs = logits.softmax(1)[0].detach()
            if class_idx is None:
                class_idx = int(logits.argmax(1))
            logits[0, class_idx].backward()  # gradient of this one class score only
        act = _to_nchw(store["act"].detach(), model)
        grad = _to_nchw(store["grad"].detach(), model)
        weights = grad.mean(dim=(2, 3), keepdim=True)        # importance of each channel for the class
        cam = F.relu((weights * act).sum(1, keepdim=True))  # weighted sum of channels; keep positive evidence
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # scale to 0..1 for the heatmap
        return cam.cpu().numpy(), class_idx, probs.cpu()
    finally:
        h1.remove()
        h2.remove()


def overlay(img_rgb, cam, alpha=0.45):
    """Blend a heatmap (jet colormap) onto an RGB uint8 image of the same size."""
    import matplotlib.cm as cm

    heat = (cm.jet(cam)[..., :3] * 255).astype(np.uint8)
    return (img_rgb * (1 - alpha) + heat * alpha).astype(np.uint8)
