"""Run several experiment configs one after another (unattended), one log per run.

Usage
    .venv/Scripts/python run_queue.py configs/e3_convnext_tiny.yaml configs/e3_swin_tiny.yaml
    .venv/Scripts/python run_queue.py configs/e3_*.yaml --skip-done

Each run's output (with live progress bars) is shown in this terminal and saved to runs/<exp_id>.log.
A failed run is reported and the queue moves on to the next config.
"""
import argparse
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
NL, CR = "\n", "\r"


def exp_id_of(cfg_path):
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f).get("exp_id")


def run_streaming(cmd, log_path):
    """Run a child process, showing its output live (progress bars included) in this terminal,
    while the log file only keeps finished lines (the final state of each progress bar)."""
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
    pending = ""
    with open(log_path, "w", encoding="utf-8") as log:
        while True:
            chunk = os.read(proc.stdout.fileno(), 4096)
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            sys.stdout.write(text)
            sys.stdout.flush()
            pending += text
            *lines, pending = pending.split(NL)
            for line in lines:
                # Windows ends lines with CR+LF; after that, the text after the last CR is
                # what a progress bar finally showed
                log.write(line.rstrip(CR).split(CR)[-1] + NL)
            log.flush()
        if pending:
            log.write(pending.rstrip(CR).split(CR)[-1] + NL)
    return proc.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--skip-done", action="store_true", help="skip configs that already have results/<id>/metrics.json")
    args = ap.parse_args()

    paths = []
    for c in args.configs:
        paths += sorted(glob.glob(c)) or [c]

    (ROOT / "runs").mkdir(exist_ok=True)
    for p in paths:
        eid = exp_id_of(p)
        if args.skip_done and (ROOT / "results" / eid / "metrics.json").exists():
            print(f"skip {eid} (done)", flush=True)
            continue
        log = ROOT / "runs" / f"{eid}.log"
        print(f"=== {eid}  ({p})  log: {log}", flush=True)
        t = time.time()
        rc = run_streaming([sys.executable, "-u", "train.py", "--config", p], log)
        status = "OK" if rc == 0 else f"FAILED rc={rc}"
        tail = log.read_text(encoding="utf-8", errors="ignore").strip().splitlines()[-1:]
        print(f"=== {eid} {status} in {(time.time()-t)/60:.1f} min | {tail[0] if tail else ''}", flush=True)


if __name__ == "__main__":
    main()
