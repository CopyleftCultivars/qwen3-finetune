"""
push_to_hub.py

Copies the generated merged_sharegpt.jsonl into this folder and pushes
the entire hf_dataset/ directory to your HuggingFace Datasets organization.

Usage:
  1. Run prepare_dataset.py first to generate merged_sharegpt.jsonl
  2. Set ORG and REPO_NAME below (or pass as env vars)
  3. python hf_dataset/push_to_hub.py

Prerequisites:
  pip install huggingface_hub
  huggingface-cli login   (or set HF_TOKEN env var)
"""

import os
import shutil
from pathlib import Path
from huggingface_hub import HfApi, login

# ── CONFIG ────────────────────────────────────────────────────────────────
ORG       = os.getenv("HF_ORG",  "[LOCAL_FOLDER]")   # e.g. "[LOCAL_FOLDER]"
REPO_NAME = os.getenv("HF_REPO", "knf-sharegpt")
HF_TOKEN  = os.getenv("HF_TOKEN", None)              # or set via huggingface-cli login
PRIVATE   = True                                      # set False to make public
# ─────────────────────────────────────────────────────────────────────────

REPO_ID   = f"{ORG}/{REPO_NAME}"
HERE      = Path(__file__).parent
JSONL_SRC = HERE.parent / "merged_sharegpt.jsonl"
JSONL_DST = HERE / "train.jsonl"


def main():
    if HF_TOKEN:
        login(token=HF_TOKEN)

    # Copy the generated JSONL into the dataset folder
    if not JSONL_SRC.exists():
        print(f"ERROR: {JSONL_SRC} not found.")
        print("Run prepare_dataset.py first to generate it.")
        return

    print(f"Copying {JSONL_SRC.name} → {JSONL_DST} ...")
    shutil.copy2(JSONL_SRC, JSONL_DST)

    api = HfApi()

    # Create the repo under the org if it doesn't exist
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="dataset",
        exist_ok=True,
        private=PRIVATE,
    )
    print(f"Repo ready: https://huggingface.co/datasets/{REPO_ID}")

    # Upload entire hf_dataset/ folder
    api.upload_folder(
        folder_path=str(HERE),
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="Add merged KNF ShareGPT training dataset",
    )

    print(f"\nDone! Dataset live at: https://huggingface.co/datasets/{REPO_ID}")
    print(f"\nLoad in Python:")
    print(f'  ds = load_dataset("{REPO_ID}", data_files="train.jsonl", split="train")')


if __name__ == "__main__":
    main()
