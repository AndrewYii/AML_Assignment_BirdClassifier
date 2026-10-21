"""Deploy the web app (the app/ folder only) to a Hugging Face Space.

Before running: log in once with   .venv\\Scripts\\hf auth login   (paste a token with WRITE access).

Usage
    .venv/Scripts/python deploy_to_hf.py --space <your-hf-username>/bird-species-identifier

What it uploads: app/ only (app.py, core/, weights/, examples/, class_names.json, models.json,
requirements.txt, README.md). Dataset, training runs, results, drafts and the virtual env stay on your PC.
The weight file (~170 MB) is stored with Git LFS automatically by huggingface_hub.
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi

APP = Path(__file__).resolve().parent / "app"
NEEDED = ["app.py", "README.md", "requirements.txt", "class_names.json", "models.json",
          "core/models.py", "core/gradcam.py"] + [f"weights/fold{i}.pt" for i in range(1, 6)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="<username>/<space-name>")
    ap.add_argument("--private", action="store_true", help="create the Space as private")
    args = ap.parse_args()

    missing = [f for f in NEEDED if not (APP / f).exists()]
    if missing:
        raise SystemExit(f"app/ is incomplete, missing: {missing} (run export_for_app.py first)")

    api = HfApi()
    print("logged in as:", api.whoami()["name"])
    api.create_repo(args.space, repo_type="space", space_sdk="gradio", private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(APP),
        repo_id=args.space,
        repo_type="space",
        ignore_patterns=["__pycache__/*", "*.pyc", "flagged/*", ".gradio/*"],
        # remove files that are no longer part of the app (e.g. older weights); current ones are re-added
        delete_patterns=["weights/*", "core/*"],
        commit_message="Deploy bird species identifier",
    )
    print(f"done: https://huggingface.co/spaces/{args.space}")
    print("The Space now installs packages and starts (a few minutes). Watch the 'Logs' tab on that page.")


if __name__ == "__main__":
    main()
