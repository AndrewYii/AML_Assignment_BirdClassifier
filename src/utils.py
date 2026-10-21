"""Small helpers shared by every script: config loading, seeding, logging, model cost.

Every experiment is described by one YAML file in ``configs/``. A config may
start with ``base: <other.yaml>`` and only list the keys it changes, so the
diff between two experiments is exactly what is written in the file. That is
what keeps the comparisons in the report fair and easy to audit.
"""
import json
import os
import random
import subprocess
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Config handling
# --------------------------------------------------------------------------- #
def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base`` (override wins)."""
    out = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path, overrides=None) -> dict:
    """Load a YAML config, resolving its ``base:`` chain, then apply CLI overrides.

    ``overrides`` is a list like ``["train.epochs=1", "data.img_size=128"]``;
    each value is parsed with YAML so numbers / booleans / null work naturally.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "base" in cfg:
        parent = load_config(path.parent / cfg.pop("base"))
        cfg = deep_merge(parent, cfg)
    for item in overrides or []:
        key, value = item.split("=", 1)
        node = cfg
        *parents, last = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[last] = yaml.safe_load(value)
    return cfg


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int):
    """Seed python, numpy and torch so runs are repeatable (up to cuDNN noise)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_hash() -> str:
    """Current commit, stored with each run so results can be traced to code."""
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------- #
# Small I/O helpers
# --------------------------------------------------------------------------- #
def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_yaml(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


class Timer:
    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *a):
        self.elapsed = time.time() - self.t


# --------------------------------------------------------------------------- #
# Model cost (reported next to accuracy so comparisons stay honest)
# --------------------------------------------------------------------------- #
def count_params(model) -> float:
    """Number of parameters in millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6


@torch.no_grad()
def measure_flops_and_latency(model, img_size, device):
    """GFLOPs for one image and average GPU latency (ms/img, batch 1, fp16)."""
    from torch.utils.flop_counter import FlopCounterMode

    model.eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)
    gflops = None
    try:
        with FlopCounterMode(display=False) as fc:
            model(x)
        # FlopCounter counts multiply and add separately -> /2 gives MACs ~ "FLOPs" as usually quoted
        gflops = fc.get_total_flops() / 2 / 1e9
    except Exception:
        pass
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        for _ in range(5):  # warm-up
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        n = 20
        for _ in range(n):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return gflops, (time.time() - t) / n * 1000
