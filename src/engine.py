"""Training scheme: optimiser, LR schedule, loss, train loop and evaluation loop.

Everything that can change between experiments is read from ``cfg['train']``:

  optimizer        adamw | adam | sgd              (E4)
  lr / head_lr     backbone LR / LR of the new 200-way head ("differential LR")
  strategy         full | linear_probe | uniform | layer_decay   (E8)
  weight_decay     L2 / decoupled weight decay       (E6)
  label_smoothing, mixup, cutmix                    (E6)
  batch_size, accum_steps                           (E5, memory)
  epochs, warmup_epochs                             fixed budget, cosine schedule
"""
import math
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.optim import param_groups_layer_decay
from timm.utils import ModelEmaV3
from tqdm import tqdm

from .metrics import compute_metrics
from .models import head_param_names


# --------------------------------------------------------------------------- #
# Optimiser
# --------------------------------------------------------------------------- #
def _no_decay(name, param, skip):
    """Biases, norm weights and position/class tokens are not weight-decayed (standard practice)."""
    return param.ndim <= 1 or name.endswith(".bias") or any(s in name for s in skip)


def build_optimizer(model, tr):
    """Create the optimiser with per-group learning rates.

    * full         : backbone at ``lr``, new head at ``head_lr`` (differential LR).
                     The pretrained backbone moves slowly so ImageNet features are not destroyed,
                     while the random head learns quickly.
    * uniform      : every parameter at ``lr`` (no differential LR, E8 ablation).
    * linear_probe : backbone frozen, only the head is trained (E8 ablation).
    * layer_decay  : timm layer-wise LR decay: head at ``head_lr``, each earlier block
                     multiplied by ``layer_decay`` (E8).
    """
    strategy = tr.get("strategy", "full")
    wd = tr["weight_decay"]
    skip = set(getattr(model, "no_weight_decay", lambda: set())())
    head_names = head_param_names(model)

    if strategy == "layer_decay":
        groups = param_groups_layer_decay(model, weight_decay=wd, no_weight_decay_list=skip,
                                          layer_decay=tr["layer_decay"])
        for g in groups:
            g["lr"] = tr["head_lr"] * g.pop("lr_scale", 1.0)
    else:
        if strategy == "linear_probe":
            for n, p in model.named_parameters():
                p.requires_grad = n in head_names
        head_lr = tr["lr"] if strategy == "uniform" else tr["head_lr"]
        buckets = {}
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            is_head = n in head_names
            key = (is_head, _no_decay(n, p, skip))
            buckets.setdefault(key, []).append(p)
        groups = [{"params": ps,
                   "lr": head_lr if is_head else tr["lr"],
                   "weight_decay": 0.0 if nd else wd}
                  for (is_head, nd), ps in buckets.items()]

    opt = tr["optimizer"].lower()
    if opt == "adamw":
        return torch.optim.AdamW(groups, betas=(0.9, 0.999))
    if opt == "adam":
        return torch.optim.Adam(groups, betas=(0.9, 0.999))  # coupled L2 weight decay
    if opt == "sgd":
        return torch.optim.SGD(groups, momentum=tr.get("momentum", 0.9), nesterov=True)
    raise ValueError(opt)


def build_scheduler(optimizer, tr, steps_per_epoch):
    """Linear warm-up then cosine decay to ``min_lr_ratio`` x initial LR, updated every step."""
    total = tr["epochs"] * steps_per_epoch
    warm = tr.get("warmup_epochs", 0) * steps_per_epoch
    floor = tr.get("min_lr_ratio", 0.01)

    def factor(step):
        if step < warm:
            return (step + 1) / warm
        progress = (step - warm) / max(1, total - warm)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


# --------------------------------------------------------------------------- #
# Train / evaluate loops
# --------------------------------------------------------------------------- #
def progress(iterable, desc, leave=True):
    """Live progress bar (steps, speed, ETA). Refreshes at most once a second so log files stay small."""
    return tqdm(iterable, desc=desc, leave=leave, ncols=110, mininterval=1.0, file=sys.stderr,
                bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}")


def train_one_epoch(model, loader, optimizer, scheduler, tr, num_classes, device, desc="train", ema=None):
    """One pass over the training set. Returns (mean loss, train accuracy).
    ``ema`` (optional): a weight-averaged copy of the model, updated after every optimiser step."""
    model.train()
    mix = None
    if tr.get("mixup", 0) or tr.get("cutmix", 0):
        # MixUp blends two images + labels; CutMix pastes a patch of one image into another.
        mix = Mixup(mixup_alpha=tr.get("mixup", 0), cutmix_alpha=tr.get("cutmix", 0),
                    prob=tr.get("mix_prob", 1.0), label_smoothing=tr.get("label_smoothing", 0.0),
                    num_classes=num_classes)
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=tr.get("label_smoothing", 0.0))

    accum = tr.get("accum_steps", 1)
    amp = tr.get("amp", True) and device.type == "cuda"
    tot_loss, correct, seen = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)
    bar = progress(loader, desc)
    for i, (x, y) in enumerate(bar):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        y_hard = y
        if mix is not None:
            x, y = mix(x, y)  # y becomes a soft label distribution
        # bfloat16 autocast: ~2x faster on RTX 30xx, no loss scaling needed
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(x)
            loss = criterion(logits, y)
        (loss / accum).backward()
        if (i + 1) % accum == 0 or (i + 1) == len(loader):
            if tr.get("clip_grad"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), tr["clip_grad"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if ema is not None:
                ema.update(model, step=ema.n_updates)
                ema.n_updates += 1
        tot_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y_hard).sum().item()
        seen += x.size(0)
        if i % 10 == 0 or i + 1 == len(loader):  # running averages shown on the bar
            bar.set_postfix_str(f"loss {tot_loss / seen:.3f} acc {correct / seen:.3f} "
                                f"lr {optimizer.param_groups[0]['lr']:.1e}")
    bar.close()
    return tot_loss / seen, correct / seen


@torch.no_grad()
def predict(model, loader, device, tta=False, amp=True, desc="test"):
    """Return (logits [N, C], targets [N]) for a loader. ``tta`` adds a horizontal-flip pass."""
    model.eval()
    all_logits, all_targets = [], []
    for x, y in progress(loader, desc, leave=False):
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            out = model(x).float()
            if tta:
                out = (out + model(torch.flip(x, dims=[3])).float()) / 2
        all_logits.append(out.cpu())
        all_targets.append(y)
    return torch.cat(all_logits).numpy(), torch.cat(all_targets).numpy()


def run_training(model, train_loader, test_loader, cfg, num_classes, device, log_path=None):
    """Full fixed-budget training. Test metrics are logged every epoch for the
    learning curves only; the reported result is always the *last* epoch.

    ``train.freeze_epochs`` > 0 gives a two-phase ("progressive") fine-tune (E14):
      phase 1: backbone frozen, only the new head is trained for ``freeze_epochs`` epochs
      phase 2: everything unfrozen for the remaining epochs, with a fresh warm-up + cosine schedule
    The total number of epochs stays ``train.epochs``.
    """
    tr = cfg["train"]
    total = tr["epochs"]
    k = tr.get("freeze_epochs", 0)
    phases = [(k, "linear_probe"), (total - k, tr.get("strategy", "full"))] if k else [(total, tr.get("strategy", "full"))]
    steps_per_epoch = math.ceil(len(train_loader) / tr.get("accum_steps", 1))

    # Exponential moving average of the weights (E15): evaluating this slowly-updated copy instead of the raw
    # weights averages out step-to-step noise -> smoother test curves and usually slightly better accuracy.
    # use_warmup ramps the averaging up (decay = 1 - (1+step)^-2/3, capped at ema_decay) so early epochs are not
    # dominated by the random initial head.
    ema = None
    if tr.get("ema_decay"):
        ema = ModelEmaV3(model, decay=tr["ema_decay"], use_warmup=True)
        ema.n_updates = 0
    eval_model = ema.module if ema is not None else model

    history, epoch = [], 0
    for n_epochs, strategy in phases:
        phase_tr = dict(tr, epochs=n_epochs, strategy=strategy)
        if strategy != "linear_probe":
            for p in model.parameters():  # (re-)enable gradients after a frozen phase
                p.requires_grad = True
        optimizer = build_optimizer(model, phase_tr)
        scheduler = build_scheduler(optimizer, phase_tr, steps_per_epoch)
        for _ in range(n_epochs):
            epoch += 1
            t0 = time.time()
            tag = f"[{cfg['exp_id']}] ep {epoch}/{total}"
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scheduler, phase_tr, num_classes,
                                              device, desc=f"{tag} train", ema=ema)
            frozen = strategy == "linear_probe"
            row = {"epoch": epoch, "phase": strategy,
                   "lr_backbone": 0.0 if frozen else min(g["lr"] for g in optimizer.param_groups),
                   "lr_head": max(g["lr"] for g in optimizer.param_groups),
                   "train_loss": tr_loss, "train_acc": tr_acc}
            if epoch % tr.get("eval_every", 1) == 0 or epoch == total:
                logits, targets = predict(eval_model, test_loader, device, desc=f"{tag} test ")
                row["test_loss"] = float(F.cross_entropy(torch.from_numpy(logits), torch.from_numpy(targets)))
                summ, _, _ = compute_metrics(logits, targets)
                row["test_top1"], row["test_mca"] = summ["top1"], summ["mean_class_acc"]
            row["epoch_time_s"] = time.time() - t0
            history.append(row)
            print(f"[{cfg['exp_id']}] ep {epoch:3d}/{total} "
                  f"loss {tr_loss:.3f} train_acc {tr_acc:.3f} "
                  f"test_top1 {row.get('test_top1', float('nan')):.4f} mca {row.get('test_mca', float('nan')):.4f} "
                  f"({row['epoch_time_s']:.0f}s)", flush=True)
            if log_path is not None:
                pd.DataFrame(history).to_csv(log_path, index=False)  # rewrite each epoch so progress is visible
    return history, eval_model


def save_evaluation(logits, targets, items, class_names, out_dir):
    """Write metrics.json, per_class.csv, preds.csv and confusion.npy for one model."""
    from .utils import save_json

    out_dir.mkdir(parents=True, exist_ok=True)
    summary, rows, cm = compute_metrics(logits, targets, class_names)
    probs = torch.softmax(torch.from_numpy(logits), 1).numpy()
    top5 = np.argsort(-logits, axis=1)[:, :5]
    pd.DataFrame(rows).to_csv(out_dir / "per_class.csv", index=False)
    pd.DataFrame({
        "image": [n for n, _ in items],
        "target": targets,
        "pred": logits.argmax(1),
        "confidence": probs.max(1),
        "top5": [" ".join(map(str, t)) for t in top5],
    }).to_csv(out_dir / "preds.csv", index=False)
    np.save(out_dir / "confusion.npy", cm)
    save_json(summary, out_dir / "metrics.json")
    return summary
