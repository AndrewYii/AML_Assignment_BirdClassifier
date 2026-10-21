"""Model factory: every architecture comes from ``timm`` with pretrained weights.

The project is transfer-learning only. All backbones (ResNet-50, ConvNeXt,
Swin, DeiT/ViT) are loaded the same way, and their 1000-class ImageNet
head is replaced by a new 200-class linear layer. Using one factory for every
model means the only difference between two runs is the backbone name, which
keeps the architecture comparison fair.
"""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import SelectAdaptivePool2d


# --------------------------------------------------------------------------- #
# Extra global pooling operators for the head-pooling experiment (E7)
# --------------------------------------------------------------------------- #
class GlobalMinPool(nn.Module):
    """Global *min* pooling: min over H x W, written as -maxpool(-x).

    Idea being tested: the strongest response at a location might be noise,
    while min pooling keeps only what is present everywhere. ConvNeXt features
    are not ReLU outputs (they can be negative), so min pooling is not trivially 0.
    """

    def forward(self, x):
        return -F.adaptive_max_pool2d(-x, 1)


class GeM(nn.Module):
    """Generalised-mean pooling (p=3 learnable). p=1 is average, p -> inf is max."""

    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), 1).pow(1.0 / self.p)


def _swap_global_pool(model, pooling):
    """Replace the pooling op inside timm's SelectAdaptivePool2d (CNN heads only)."""
    if pooling in (None, "avg"):
        return
    new = {"max": lambda: nn.AdaptiveMaxPool2d(1), "min": GlobalMinPool, "gem": GeM}[pooling]
    swapped = 0
    for m in model.modules():
        if isinstance(m, SelectAdaptivePool2d):
            m.pool = new()
            swapped += 1
    if swapped == 0:
        raise ValueError(f"pooling='{pooling}' only supported for CNN backbones with a global pool")


def _is_transformer(name):
    return any(k in name for k in ("vit", "deit", "swin", "eva", "beit"))


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_model(cfg, num_classes):
    """Create the network described by ``cfg['model']``.

    Keys used
      name        timm model name incl. pretrained tag, e.g. 'convnext_tiny.fb_in1k'
      pretrained  load pretrained weights (always true in this project)
      dropout     dropout before the classifier
      drop_path   stochastic depth rate (regularisation, E6)
      pooling     avg | max | min | gem   (E7, CNNs only)
      grad_ckpt   gradient checkpointing to fit big models in 8 GB
    """
    m = cfg["model"]
    kwargs = dict(
        pretrained=m.get("pretrained", True),
        num_classes=num_classes,  # new randomly initialised 200-way head
        drop_rate=m.get("dropout", 0.0),
        drop_path_rate=m.get("drop_path", 0.0),
    )
    if _is_transformer(m["name"]):
        # transformers have a fixed patch grid; tell timm the training resolution so it
        # interpolates the position embeddings if the resolution differs from pretraining
        kwargs["img_size"] = cfg["data"]["img_size"]
    model = timm.create_model(m["name"], **kwargs)
    _swap_global_pool(model, m.get("pooling", "avg"))
    if m.get("grad_ckpt"):
        model.set_grad_checkpointing(True)
    return model


def norm_stats(model):
    """Mean/std the backbone was pretrained with (read from its timm pretrained config)."""
    pc = model.pretrained_cfg
    return pc.get("mean", (0.485, 0.456, 0.406)), pc.get("std", (0.229, 0.224, 0.225))


def head_param_names(model):
    """Names of the parameters in the new classifier layer (they get the higher head LR)."""
    prefix = model.pretrained_cfg.get("classifier", "head")
    if isinstance(prefix, (list, tuple)):
        prefix = prefix[0]
    return {n for n, _ in model.named_parameters() if n == prefix or n.startswith(prefix + ".")}


@torch.no_grad()
def extract_features(model, x):
    """Penultimate (pre-logit) embedding, used for t-SNE."""
    return model.forward_head(model.forward_features(x), pre_logits=True)
