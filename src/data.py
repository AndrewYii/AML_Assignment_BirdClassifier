"""Data input, pre-processing and data loaders for the CUB-200 assignment split.

Annotation files (``dataset/train.txt`` / ``dataset/test.txt``) have one line per
image: ``<file name> <class id>``. Training images are named after the species
(``Black_footed_Albatross_0004_2731401028.jpg``), so the class names are
recovered from the training file names.

There is deliberately **no validation split**: the training set has only about
24 images per class, so every image is used for training. Instead of picking
the "best" epoch, every model trains for a fixed number of epochs and we report
the last one; 5-fold cross-validation on the training set checks the result.
"""
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .utils import ROOT


# --------------------------------------------------------------------------- #
# Annotation parsing
# --------------------------------------------------------------------------- #
def read_split(txt_path):
    """Return a list of (file_name, label) tuples from an annotation file."""
    items = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, label = line.rsplit(" ", 1)
            items.append((name, int(label)))
    return items


def class_names_from_train(train_txt):
    """Map class id -> readable species name, e.g. 0 -> 'Black footed Albatross'.

    The species is everything before the last two '_' fields (image id, flickr id).
    """
    names = {}
    for fname, label in read_split(train_txt):
        species = Path(fname).stem.rsplit("_", 2)[0].replace("_", " ")
        names.setdefault(label, species)
    return [names[i] for i in range(len(names))]


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class CUBDataset(Dataset):
    """Loads an image from disk, converts to RGB and applies a transform."""

    def __init__(self, items, img_dir, transform=None):
        self.items = items
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.targets = [y for _, y in items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        name, label = self.items[idx]
        img = Image.open(self.img_dir / name).convert("RGB")  # a few CUB images are greyscale
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def make_subset(items, subset_cfg, seed):
    """Optionally shrink the training set (used by the experiments, never on test).

    * ``{n_classes: 10, per_class: 5}`` -> tiny subset for the overfit sanity check (E1).
      Labels are re-mapped to 0..n_classes-1.
    * ``{fraction: 0.25}`` -> keep that fraction of every class (stratified, E9).
    """
    if not subset_cfg:
        return items, None
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for name, y in items:
        by_class[y].append((name, y))

    if "n_classes" in subset_cfg:
        classes = sorted(by_class)[: subset_cfg["n_classes"]]
        remap = {c: i for i, c in enumerate(classes)}
        out = []
        for c in classes:
            picked = rng.sample(by_class[c], min(subset_cfg["per_class"], len(by_class[c])))
            out += [(n, remap[c]) for n, _ in picked]
        return out, remap

    frac = subset_cfg["fraction"]
    out = []
    for c in sorted(by_class):
        k = max(1, math.ceil(frac * len(by_class[c])))  # at least 1 image per class
        out += rng.sample(by_class[c], k)
    return out, None


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def build_transforms(img_size, aug, mean, std, crop_pct=0.875):
    """Return (train_transform, test_transform).

    The test transform is identical for every experiment: resize the short side
    to ``img_size / crop_pct`` then centre-crop ``img_size``. Only the training
    augmentation level changes between experiments.

    aug levels
      none   : same as test (used for the overfit sanity check)
      basic  : RandomResizedCrop + flip + small rotation + colour jitter (baseline)
      strong : basic + RandAugment + RandomErasing (regularisation ladder, E6)
    """
    resize = int(math.floor(img_size / crop_pct))
    normalize = T.Normalize(mean, std)
    test_tf = T.Compose([
        T.Resize(resize, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        normalize,
    ])
    if aug == "none":
        return test_tf, test_tf

    train_ops = [
        # random crop covering 35-100% of the image: birds are often small in the frame,
        # so we do not crop more aggressively than this
        T.RandomResizedCrop(img_size, scale=(0.35, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(0.2, 0.2, 0.2),
    ]
    if aug == "strong":
        train_ops.append(T.RandAugment(num_ops=2, magnitude=9))
    train_ops += [T.ToTensor(), normalize]
    if aug == "strong":
        train_ops.append(T.RandomErasing(p=0.25))  # simulates occlusion by branches/leaves
    return T.Compose(train_ops), test_tf


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def build_loaders(cfg, mean, std):
    """Build train / test DataLoaders from the ``data`` and ``train`` config sections."""
    d, tr = cfg["data"], cfg["train"]
    train_items = read_split(ROOT / d["train_txt"])
    test_items = read_split(ROOT / d["test_txt"])

    train_items, remap = make_subset(train_items, d.get("subset"), cfg["seed"])
    if remap is not None:
        # tiny-subset sanity check: evaluate on the *training* images themselves
        # (the goal is to prove the model can memorise, not to generalise)
        test_items, test_dir = train_items, d["train_dir"]
    else:
        test_dir = d["test_dir"]

    train_tf, test_tf = build_transforms(d["img_size"], d["aug"], mean, std, d.get("crop_pct", 0.875))
    train_ds = CUBDataset(train_items, ROOT / d["train_dir"], train_tf)
    test_ds = CUBDataset(test_items, ROOT / test_dir, test_tf)

    sampler, shuffle = None, True
    if tr.get("balanced_sampler"):
        # optional (E6b): sample rare classes as often as common ones
        counts = torch.bincount(torch.tensor(train_ds.targets))
        weights = (1.0 / counts.float())[train_ds.targets]
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)
        shuffle = False

    # Each Windows worker is a full Python process (~0.5 GB RAM), so worker counts are kept modest:
    # the test set is small and only needs 2 workers.
    nw = d.get("num_workers", 4)
    nw_test = min(2, nw)
    train_loader = DataLoader(train_ds, batch_size=tr["batch_size"], shuffle=shuffle, sampler=sampler,
                              # timm's mixup/cutmix needs an even batch size -> drop the ragged last batch
                              drop_last=bool(tr.get("mixup", 0) or tr.get("cutmix", 0)),
                              num_workers=nw, pin_memory=True, persistent_workers=nw > 0)
    test_loader = DataLoader(test_ds, batch_size=tr.get("eval_batch_size", 64), shuffle=False,
                             num_workers=nw_test, pin_memory=True, persistent_workers=nw_test > 0)
    n_classes = len(remap) if remap is not None else cfg["model"]["num_classes"]
    return train_loader, test_loader, n_classes
